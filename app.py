from hashlib import sha256
import time

import streamlit as st

from rag import (
    ROOT, Settings, ask, build_knowledge_base, document_identity,
    friendly_error, read_defaults, read_documents, source_location, split_documents,
)

st.set_page_config(page_title="资料问答助手", layout="wide")
st.title("资料问答助手")
st.caption("读懂一份资料，从一个问题开始。回答附带本次检索到的参考片段。")

defaults = read_defaults()
with st.sidebar:
    st.header("模型配置")
    api_key = st.text_input(
        "百炼 API Key", type="password", value=defaults["DASHSCOPE_API_KEY"],
        help="可在此填写，或配置到本地 .env；页面填写的 Key 不写入文件。",
    ).strip()
    with st.expander("模型与接口"):
        chat_model = st.text_input("回答模型", defaults["CHAT_MODEL"]).strip()
        embedding_model = st.text_input("向量模型", defaults["EMBEDDING_MODEL"]).strip()
        base_url = st.text_input("百炼兼容接口地址", defaults["DASHSCOPE_BASE_URL"]).strip()
    st.caption("处理资料和提问会调用百炼，消耗相应模型额度；资料片段会发送至百炼。")
    if st.button("清除当前资料与问答", use_container_width=True):
        for key in ("knowledge", "history", "active_material", "upload_token"):
            st.session_state.pop(key, None)
        st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
        st.rerun()
    st.divider()
    st.markdown("**使用步骤**\n\n① 填写 Key\n\n② 上传资料并处理\n\n③ 输入问题，核对参考片段")
    st.caption("仅供本机课程演示。关闭会话后，内存索引不再保留。")

settings = Settings(api_key, base_url, chat_model, embedding_model)
left, right = st.columns([1, 1.8], gap="large")
with left:
    st.subheader("1 · 准备资料")
    uploaded = st.file_uploader(
        "上传 PDF、TXT 或 Markdown",
        type=["pdf", "txt", "md"],
        key=f"upload_{st.session_state.get('uploader_version', 0)}",
        help="PDF 需要包含可提取的文字；最大 10 MB、100 页。不支持扫描图片识别。",
    )
    if uploaded is not None:
        token = sha256(uploaded.name.encode() + uploaded.getvalue()).hexdigest()
        if token != st.session_state.get("upload_token"):
            st.session_state["active_material"] = (uploaded.name, uploaded.getvalue())
            st.session_state["upload_token"] = token
    elif st.session_state.pop("upload_token", None) is not None:
        st.session_state.pop("active_material", None)
    if st.button("使用示例资料", use_container_width=True):
        sample = ROOT / "examples" / "示例课程资料.txt"
        st.session_state["active_material"] = (sample.name, sample.read_bytes())

    material = st.session_state.get("active_material")
    if material:
        filename, content = material
        st.write(f"当前资料：**{filename}**")
        try:
            documents = read_documents(filename, content)
            chunks = split_documents(documents)
            st.caption(
                f"{len(content) / 1024:.1f} KB · "
                f"{sum(len(d.page_content) for d in documents):,} 字符 · {len(chunks)} 个片段"
            )
            with st.expander("预览提取的文字"):
                st.text(documents[0].page_content[:2000])
            if st.button("处理资料", type="primary", use_container_width=True):
                with st.spinner("正在建立资料索引……"):
                    try:
                        # 新索引完整建立后再替换旧状态，失败时不留下半成品。
                        knowledge = build_knowledge_base(filename, content, settings)
                        st.session_state["knowledge"] = knowledge
                        st.session_state["history"] = []
                        st.success("资料已处理，可以开始提问。")
                    except Exception as error:
                        st.error(friendly_error(error))
        except Exception as error:
            st.error(friendly_error(error))
    else:
        st.info("上传一份资料，或先使用自带示例。")

with right:
    st.subheader("2 · 根据资料提问")
    knowledge = st.session_state.get("knowledge")
    ready = bool(
        knowledge and material
        and knowledge.identity == document_identity(material[0], material[1], settings)
    )
    if knowledge and not ready:
        st.warning("资料、Key 或向量配置已变化，请重新处理当前资料后再提问。")
    if ready:
        st.success(f"已就绪：{knowledge.filename}")
        history = st.session_state.get("history", [])
        for item in history:
            with st.container(border=True):
                st.markdown("**问题**")
                st.write(item["question"])
            with st.container(border=True):
                st.markdown("**回答**")
                st.write(item["text"])
                if item["citation_ids"]:
                    st.caption("回答引用：" + " ".join(f"[{i}]" for i in item["citation_ids"]))
                with st.expander("查看本次检索的参考片段"):
                    for source in item["sources"]:
                        st.markdown(f"**[{source['id']}] {source['source']} · {source['location']}**")
                        st.text(source["text"])
                st.caption(f"本次耗时 {item['seconds']:.1f} 秒")
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "你的问题", max_chars=1000,
                placeholder="例如：事务的原子性是什么意思？", height=100,
            )
            submitted = st.form_submit_button("提问", type="primary")
        if submitted:
            started = time.perf_counter()
            with st.spinner("正在检索资料并生成回答……"):
                try:
                    answer = ask(knowledge, question, settings)
                    item = {
                        "question": question.strip(), "text": answer.text,
                        "citation_ids": answer.citation_ids,
                        "sources": [
                            {"id": i, "source": d.metadata["source"],
                             "location": source_location(d), "text": d.page_content}
                            for i, d in enumerate(answer.sources, 1)
                        ],
                        "seconds": time.perf_counter() - started,
                    }
                    st.session_state["history"] = (history + [item])[-12:]
                    st.rerun()
                except Exception as error:
                    st.error(friendly_error(error))
    else:
        st.info("资料处理完成后，这里会出现提问入口。")
        st.markdown("你可以问：\n\n- 这个概念是什么意思？\n- 资料中提到了哪些处理步骤？\n- 两种方案有什么区别？")
    st.caption("回答可能出错，请结合参考片段核对。每次提问独立检索，不使用之前的问答作为资料。")
