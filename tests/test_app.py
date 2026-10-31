import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

import rag


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count(word)) for word in ("事务", "原子", "缓存", "索引", "权限")] + [1.0]


SETTINGS = rag.Settings("local-test-placeholder")
DEFAULTS = {
    "DASHSCOPE_API_KEY": "", "DASHSCOPE_BASE_URL": SETTINGS.base_url,
    "CHAT_MODEL": SETTINGS.chat_model, "EMBEDDING_MODEL": SETTINGS.embedding_model,
}


def local_chain(supported=True, citations=None):
    response = json.dumps({
        "supported": supported, "answer": "事务操作应作为整体提交或回滚。",
        "citation_ids": [1] if citations is None else citations,
    }, ensure_ascii=False)
    with patch("rag.ChatOpenAI", return_value=FakeListChatModel(responses=[response])):
        return rag.make_answer_chain(SETTINGS)


class ReadingTests(unittest.TestCase):
    def test_utf8_bom_and_source_name(self):
        docs = rag.read_documents(r"C:\private\notes.txt", b"\xef\xbb\xbfhello")
        self.assertEqual(docs[0].page_content, "hello")
        self.assertEqual(docs[0].metadata["source"], "notes.txt")

    def test_reject_empty_and_unsupported_and_wrong_encoding(self):
        for name, content in [
            ("a.txt", b""), ("a.docx", b"word"), ("a.txt", b"\xff\xfe"),
            ("a.txt", b" " * 20), ("a.pdf", b"not a PDF"),
        ]:
            with self.subTest(name=name, content=content[:10]):
                with self.assertRaises(rag.UserError):
                    rag.read_documents(name, content)

    def test_file_size_and_text_limit(self):
        with self.assertRaises(rag.UserError):
            rag.read_documents("a.txt", b"x" * (rag.MAX_FILE_BYTES + 1))
        with self.assertRaises(rag.UserError):
            rag.read_documents("a.txt", b"x" * (rag.MAX_CHARACTERS + 1))

    def test_blank_scanned_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(rag.UserError, "OCR"):
            rag.read_documents("scan.pdf", data.getvalue())

    def test_text_pdf_preserves_page_number(self):
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=400, height=400)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 350 Td (Atomicity means all or nothing.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        data = io.BytesIO()
        writer.write(data)
        docs = rag.read_documents('notes.pdf', data.getvalue())
        self.assertIn('Atomicity', docs[0].page_content)
        self.assertEqual(docs[0].metadata['page'], 1)

    def test_pdf_page_limit(self):
        writer = PdfWriter()
        for _ in range(rag.MAX_PAGES + 1):
            writer.add_blank_page(width=10, height=10)
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(rag.UserError, "100"):
            rag.read_documents("long.pdf", data.getvalue())

    def test_split_preserves_metadata_and_limits_length(self):
        docs = rag.read_documents("a.txt", ("事务" * 1200).encode())
        chunks = rag.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(d.page_content) <= 800 for d in chunks))
        self.assertEqual(chunks[1].metadata["chunk"], 2)
        self.assertEqual(chunks[1].metadata["source"], "a.txt")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = rag.build_knowledge_base(
            "notes.txt", "事务的原子性是整体提交或回滚。缓存读取先查缓存。".encode(),
            SETTINGS, embeddings=LocalEmbeddings(),
        )

    def test_grounded_answer_with_real_chain_parser(self):
        result = rag.ask(self.knowledge, "事务原子性？", SETTINGS, chain=local_chain())
        self.assertTrue(result.supported)
        self.assertEqual(result.citation_ids, [1])
        self.assertIn("原子性", result.sources[0].page_content)

    def test_missing_evidence_uses_fixed_answer(self):
        result = rag.ask(self.knowledge, "学费？", SETTINGS, chain=local_chain(False))
        self.assertFalse(result.supported)
        self.assertEqual(result.text, rag.NO_EVIDENCE)
        self.assertEqual(result.citation_ids, [])

    def test_invalid_citations_rejected(self):
        for ids in ([], [0], [99]):
            with self.subTest(ids=ids), self.assertRaises(rag.UserError):
                rag.ask(self.knowledge, "事务？", SETTINGS, chain=local_chain(citations=ids))

    def test_blank_and_long_question_do_not_call_model(self):
        for question in (" ", "x" * 1001):
            with self.assertRaises(rag.UserError):
                rag.ask(self.knowledge, question, SETTINGS)

    def test_changed_embedding_config_rejected(self):
        changed = rag.Settings("different-key")
        with self.assertRaisesRegex(rag.UserError, "变化"):
            rag.ask(self.knowledge, "事务？", changed, chain=local_chain())

    def test_document_identity_changes_on_file_and_key(self):
        original = rag.document_identity("a.txt", b"a", SETTINGS)
        self.assertNotEqual(original, rag.document_identity("a.txt", b"b", SETTINGS))
        self.assertNotEqual(original, rag.document_identity("a.txt", b"a", rag.Settings("other")))

    def test_api_key_is_not_in_repr_or_generic_error(self):
        secret = "local-test-secret"
        self.assertNotIn(secret, repr(rag.Settings(secret)))
        self.assertNotIn(secret, rag.friendly_error(RuntimeError(secret)))

    def test_only_official_https_endpoint_allowed(self):
        for url in ("http://dashscope.aliyuncs.com/compatible-mode/v1",
                    "https://example.com/compatible-mode/v1",
                    "https://dashscope.aliyuncs.com.evil.test/compatible-mode/v1"):
            with self.assertRaises(rag.UserError):
                rag.Settings("key", url).validate()

    def test_embedding_payload_keeps_strings_and_batches_ten(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(200, json={
                "object": "list", "model": SETTINGS.embedding_model,
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                         for i, _ in enumerate(payload["input"])],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            with patch("rag.OpenAIEmbeddings", side_effect=lambda **kw: OpenAIEmbeddings(http_client=client, **kw)):
                embedder = rag.make_embeddings(SETTINGS)
            result = embedder.embed_documents(["事务"] * 23)
            self.assertEqual(len(result), 23)
            self.assertEqual([len(p["input"]) for p in requests], [10, 10, 3])
            self.assertTrue(all(p["encoding_format"] == "float" for p in requests))
            self.assertEqual(requests[0]["input"][0], "事务")
        finally:
            client.close()


class PageTests(unittest.TestCase):
    def test_sample_index_question_and_clear(self):
        with patch("rag.read_defaults", return_value=DEFAULTS), \
             patch("rag.make_embeddings", return_value=LocalEmbeddings()), \
             patch("rag.make_answer_chain", return_value=local_chain()):
            app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == "使用示例资料").click().run()
            self.assertTrue(any("当前资料" in m.value for m in app.markdown))
            next(b for b in app.button if b.label == "处理资料").click().run()
            self.assertTrue(any("API Key" in e.value for e in app.error))
            app.text_input[0].set_value("local-test-placeholder").run()
            next(b for b in app.button if b.label == "处理资料").click().run()
            self.assertEqual(len(app.exception), 0)
            app.text_area[0].set_value("事务的原子性？")
            next(b for b in app.button if b.label == "提问").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state["history"]), 1)
            self.assertTrue(any("回滚" in m.value for m in app.markdown))
            app.text_input[0].set_value("changed-key").run()
            self.assertEqual(len(app.text_area), 0)
            next(b for b in app.button if b.label == "清除当前资料与问答").click().run()
            self.assertEqual(len(app.text_area), 0)
            self.assertFalse(any("回滚" in m.value for m in app.markdown))
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
