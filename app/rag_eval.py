import argparse
from dataclasses import dataclass
from typing import List

from rag import (
    PAPERS_DIR,
    chunk_documents,
    load_documents,
    retrieve,
)


@dataclass
class RAGEvalCase:
    """
    RAG 检索评估样例。

    question：测试问题
    expected_doc_ids：期望命中的文档 ID，例如 ["demo.txt"]
    note：备注，说明这个问题想测什么
    """

    question: str
    expected_doc_ids: List[str]
    note: str = ""


@dataclass
class RAGEvalResult:
    """
    单条 RAG 检索评估结果。

    passed：
        top-k 是否命中期望文档。

    top1_passed：
        top-1 是否命中期望文档。
    """

    case: RAGEvalCase
    passed: bool
    top1_passed: bool
    top_k: int
    retrieved_doc_ids: List[str]
    retrieved_chunk_ids: List[str]
    hit_doc_ids: List[str]
    scores: List[float]


def get_eval_cases() -> List[RAGEvalCase]:
    """
    Day 6：最小评估集。

    当前 data/papers 下有两个样例文档：

    demo.txt：
        普通模块化 Agent framework。
        重点内容包括：
        task planning、tool execution、memory management、verification。

    agent_test.txt：
        Agentic RAG framework。
        重点内容包括：
        query planning、iterative retrieval、evidence verification、answer generation。

    评估问题要尽量和真实文档内容对应，避免把正确检索结果误判成错误。
    """
    return [
        RAGEvalCase(
            question="模块化 Agent 框架将哪些部分分离开来？",
            expected_doc_ids=["demo.txt"],
            note="测试普通 Agent framework：task planning、tool execution、memory management、verification",
        ),
        RAGEvalCase(
            question="普通 Agent 系统存在哪些问题？",
            expected_doc_ids=["demo.txt"],
            note="测试 unreliable tool selection、hallucinated reasoning、lack of evaluation",
        ),
        RAGEvalCase(
            question="Agentic RAG 框架包含哪些步骤？",
            expected_doc_ids=["agent_test.txt"],
            note="测试 query planning、iterative retrieval、evidence verification、answer generation",
        ),
        RAGEvalCase(
            question="传统 RAG 系统有什么问题？",
            expected_doc_ids=["agent_test.txt"],
            note="测试 irrelevant passages 和缺少 evidence verification 的问题",
        ),
    ]


def evaluate_retrieval_case(
    case: RAGEvalCase,
    chunks,
    top_k: int = 3,
) -> RAGEvalResult:
    """
    评估单条问题的检索结果。

    top-k 命中：
        只要 top-k 检索结果中出现 expected_doc_ids 里的任意一个文档，就算通过。

    top-1 命中：
        第一个检索结果必须属于 expected_doc_ids，才算通过。
    """
    results = retrieve(
        question=case.question,
        chunks=chunks,
        top_k=top_k,
    )

    retrieved_doc_ids = [item.chunk.doc_id for item in results]
    retrieved_chunk_ids = [item.chunk.chunk_id for item in results]
    scores = [item.score for item in results]

    expected_set = set(case.expected_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    hit_doc_ids = sorted(expected_set.intersection(retrieved_set))

    passed = len(hit_doc_ids) > 0

    top1_passed = bool(
        retrieved_doc_ids and retrieved_doc_ids[0] in expected_set
    )

    return RAGEvalResult(
        case=case,
        passed=passed,
        top1_passed=top1_passed,
        top_k=top_k,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        hit_doc_ids=hit_doc_ids,
        scores=scores,
    )


def format_eval_result(result: RAGEvalResult, index: int) -> str:
    """
    格式化单条评估结果。
    """
    status = "PASS" if result.passed else "FAIL"
    top1_status = "PASS" if result.top1_passed else "FAIL"

    lines = [
        f"[{index}] {status}",
        f"question: {result.case.question}",
        f"expected_doc_ids: {result.case.expected_doc_ids}",
        f"retrieved_doc_ids: {result.retrieved_doc_ids}",
        f"hit_doc_ids: {result.hit_doc_ids}",
        f"top_k_hit: {result.passed}",
        f"top1_hit: {result.top1_passed} ({top1_status})",
        f"top_k: {result.top_k}",
    ]

    if result.case.note:
        lines.append(f"note: {result.case.note}")

    lines.append("retrieved_chunks:")

    if not result.retrieved_chunk_ids:
        lines.append("  无检索结果。")
    else:
        for rank, (chunk_id, score) in enumerate(
            zip(result.retrieved_chunk_ids, result.scores),
            start=1,
        ):
            lines.append(f"  {rank}. score={score:.2f} | {chunk_id}")

    return "\n".join(lines)


def run_retrieval_eval(
    top_k: int = 3,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[RAGEvalResult]:
    """
    运行 RAG 检索评估。

    流程：
    1. 加载文档
    2. 切块
    3. 对每个测试问题执行 retrieve()
    4. 判断 top-k 是否命中期望文档
    5. 判断 top-1 是否命中期望文档
    """
    documents = load_documents(PAPERS_DIR)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    cases = get_eval_cases()

    results: List[RAGEvalResult] = []

    for case in cases:
        result = evaluate_retrieval_case(
            case=case,
            chunks=chunks,
            top_k=top_k,
        )
        results.append(result)

    return results


def print_eval_report(results: List[RAGEvalResult]) -> None:
    """
    打印评估报告。
    """
    total = len(results)

    top_k_passed = sum(1 for result in results if result.passed)
    top_k_failed = total - top_k_passed

    top1_passed = sum(1 for result in results if result.top1_passed)
    top1_failed = total - top1_passed

    top_k_hit_rate = top_k_passed / total if total > 0 else 0.0
    top1_hit_rate = top1_passed / total if total > 0 else 0.0

    print("RAG Retrieval Eval Report")
    print("=" * 40)
    print(f"total: {total}")
    print(f"top_k_passed: {top_k_passed}")
    print(f"top_k_failed: {top_k_failed}")
    print(f"top_k_hit_rate: {top_k_hit_rate:.2%}")
    print(f"top1_passed: {top1_passed}")
    print(f"top1_failed: {top1_failed}")
    print(f"top1_hit_rate: {top1_hit_rate:.2%}")
    print("=" * 40)
    print()

    for index, result in enumerate(results, start=1):
        print(format_eval_result(result, index=index))
        print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 6: RAG retrieval evaluation")

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="检索返回的 chunk 数量，默认 3",
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

    args = parser.parse_args()

    results = run_retrieval_eval(
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print_eval_report(results)


if __name__ == "__main__":
    main()