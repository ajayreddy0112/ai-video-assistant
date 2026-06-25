"""CLI entry point. Run the full pipeline against a URL or local file."""

from __future__ import annotations

from core.config import setup_logging
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input


def run_pipeline(source: str, language: str = "english") -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    return {
        "title": generate_title(transcript),
        "transcript": transcript,
        "summary": summarize(transcript),
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "open_questions": extract_questions(transcript),
        "rag_chain": build_rag_chain(transcript),
    }


def main() -> None:
    setup_logging()
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish) [english]: ").strip() or "english"
    result = run_pipeline(source, language)

    sep = "=" * 60
    print(f"\n{sep}\n📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}\n{sep}")

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    history: list[dict] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        out = ask_question(rag_chain, question, history)
        print(f"\n🤖 Assistant: {out['answer']}\n")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": out["answer"]})


if __name__ == "__main__":
    main()
