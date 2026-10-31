"""使用自带资料执行少量真实百炼验收；需要本地 .env。"""
import json

from rag import ROOT, Settings, ask, build_knowledge_base, friendly_error, read_defaults


def main():
    defaults = read_defaults()
    settings = Settings(
        defaults["DASHSCOPE_API_KEY"], defaults["DASHSCOPE_BASE_URL"],
        defaults["CHAT_MODEL"], defaults["EMBEDDING_MODEL"],
    )
    cases = [
        ("事务的原子性是什么意思？", True),
        ("Cache Aside 在缓存未命中时会怎么做？", True),
        ("这份资料所在学校明年的学费是多少？", False),
    ]
    try:
        sample = ROOT / "examples" / "示例课程资料.txt"
        knowledge = build_knowledge_base(sample.name, sample.read_bytes(), settings)
        results = []
        for question, expected_supported in cases:
            answer = ask(knowledge, question, settings)
            passed = answer.supported == expected_supported
            results.append({
                "question": question, "answer": answer.text,
                "supported": answer.supported, "citations": answer.citation_ids,
                "expected_supported": expected_supported, "passed": passed,
            })
            print(("PASS" if passed else "FAIL") + " " + question)
        report = {
            "chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
            "chunks": len(knowledge.chunks), "results": results,
            "passed": all(case["passed"] for case in results),
        }
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "live-check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0 if report["passed"] else 1
    except Exception as error:
        print(friendly_error(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
