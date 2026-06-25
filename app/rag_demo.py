import argparse
from pathlib import Path

from rag import (
    PAPERS_DIR,
    answer_with_rag,
    chunk_documents,
    format_chunk_summary,
    format_document_summary,
    format_rag_answer,
    format_retrieval_results,
    load_documents,
    retrieve,
)


def make_preview(text: str, max_chars: int = 180) -> str:
    """
    生成单行预览，方便确认文档或 chunk 是否正确读入。
    """
    one_line = " ".join(text.split())

    if len(one_line) <= max_chars:
        return one_line

    return one_line[:max_chars] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 1-5: RAG document loader, chunking, retrieval, answer and citation demo"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=str(PAPERS_DIR),
        help="要加载的论文文本目录，默认是 data/papers",
    )

    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="是否显示文档或 chunk 的内容预览",
    )

    parser.add_argument(
        "--chunks",
        action="store_true",
        help="是否执行 Day 2 chunking",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="每个 chunk 的目标字符数，默认 800",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="相邻 chunk 的重叠字符数，默认 120",
    )

    parser.add_argument(
        "--max-preview-chunks",
        type=int,
        default=5,
        help="最多预览多少个 chunk，默认 5",
    )

    parser.add_argument(
        "--retrieve",
        type=str,
        default="",
        help="输入一个问题，对 chunks 执行关键词检索",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="检索返回的 chunk 数量，默认 3",
    )

    parser.add_argument(
        "--answer",
        type=str,
        default="",
        help="输入一个问题，执行 RAG 检索并调用 LLM 回答",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="传给 LLM 的最大上下文字符数，默认 4000",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="RAG 回答使用的最低检索分数，默认 1.0",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="是否显示实际传给 LLM 的检索上下文",
    )

    parser.add_argument(
        "--only-cited-sources",
        action="store_true",
        help="只显示回答中实际引用到的来源",
    )

    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=240,
        help="引用来源预览的最大字符数，默认 240",
    )

    args = parser.parse_args()

    documents = load_documents(Path(args.dir))

    print(format_document_summary(documents))

    # 如果没有要求切块、检索或回答，就只展示文档加载结果。
    if not args.chunks and not args.retrieve and not args.answer:
        if args.show_preview:
            print("\n文档预览：")

            for index, doc in enumerate(documents, start=1):
                print(f"\n[{index}] {doc.doc_id}")
                print(make_preview(doc.text))

        return

    # 只要进入 chunking / retrieval / answer，都需要先切块。
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print()
    print(format_chunk_summary(chunks))

    if args.show_preview:
        print("\nChunk 预览：")

        for index, chunk in enumerate(chunks[: args.max_preview_chunks], start=1):
            print(f"\n[{index}] {chunk.chunk_id}")
            print(f"source: {chunk.source}")
            print(f"span: {chunk.start_char}-{chunk.end_char}")
            print(make_preview(chunk.text, max_chars=240))

    if args.retrieve:
        print("\n检索问题：")
        print(args.retrieve)

        results = retrieve(
            question=args.retrieve,
            chunks=chunks,
            top_k=args.top_k,
        )

        print()
        print(format_retrieval_results(results))

    if args.answer:
        print("\nRAG 问题：")
        print(args.answer)

        rag_result = answer_with_rag(
            question=args.answer,
            chunks=chunks,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            min_score=args.min_score,
        )

        print()
        print(
            format_rag_answer(
                rag_result,
                show_context=args.show_context,
                only_cited_sources=args.only_cited_sources,
                excerpt_chars=args.excerpt_chars,
            )
        )


if __name__ == "__main__":
    main()