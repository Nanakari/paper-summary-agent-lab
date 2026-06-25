import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from llm import call_llm


ROOT_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "data" / "papers"

DEFAULT_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass
class RAGDocument:
    """
    RAG 文档的最小数据结构。

    Day 1：一篇文件对应一个 RAGDocument。
    Day 2：一个 RAGDocument 会被切成多个 RAGChunk。
    """

    doc_id: str
    source: str
    path: str
    text: str
    num_chars: int
    num_lines: int


@dataclass
class RAGChunk:
    """
    RAG 检索的最小片段单位。

    后续检索、问答、引用都基于 RAGChunk，而不是直接基于整篇文档。
    """

    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    num_chars: int


@dataclass
class RetrievalResult:
    """
    检索结果。

    chunk：被检索命中的文本块
    score：简单关键词得分
    matched_terms：命中的关键词，方便调试
    """

    chunk: RAGChunk
    score: float
    matched_terms: List[str]


@dataclass
class RAGAnswerResult:
    """
    RAG 回答结果。

    question：用户问题
    answer：LLM 基于检索证据生成的回答
    retrieval_results：用于回答的检索片段
    context：实际传给 LLM 的上下文
    """

    question: str
    answer: str
    retrieval_results: List[RetrievalResult]
    context: str


@dataclass
class CitationSource:
    """
    引用来源。

    label：回答中的引用编号，例如 [1]
    chunk_id：对应的 chunk id
    source：来源文件
    span：字符范围
    score：检索分数
    matched_terms：命中的关键词
    excerpt：证据片段预览
    used_in_answer：这个编号是否真的出现在回答中
    """

    label: str
    chunk_id: str
    source: str
    span: str
    score: float
    matched_terms: List[str]
    excerpt: str
    used_in_answer: bool


def _is_relative_to(child: Path, parent: Path) -> bool:
    """
    Python 3.10 兼容版 Path.is_relative_to。

    用于确保读取路径没有逃出 data/papers。
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def normalize_text(text: str) -> str:
    """
    统一换行符，并去掉首尾空白。

    注意：
    这里不做复杂清洗，避免提前破坏论文内容。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def read_text_file(file_path: Path, base_dir: Path = PAPERS_DIR) -> str:
    """
    安全读取单个文本文件。

    安全边界：
    - 只允许读取 base_dir 目录下的文件
    - 只允许读取 .txt / .md / .markdown
    - 默认优先 UTF-8，必要时兼容 GBK
    """
    base_dir = base_dir.resolve()

    path = Path(file_path)
    if not path.is_absolute():
        path = base_dir / path

    path = path.resolve()

    if not _is_relative_to(path, base_dir):
        raise ValueError(f"非法路径：文件不在允许目录内：{path}")

    if path.suffix.lower() not in DEFAULT_EXTENSIONS:
        raise ValueError(f"不支持的文件类型：{path.suffix}")

    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    last_error = None

    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法用 utf-8 / utf-8-sig / gbk 解码文件：{path}，原始错误：{last_error}",
    )


def load_documents(
    papers_dir: Path = PAPERS_DIR,
    allowed_extensions: Sequence[str] = tuple(DEFAULT_EXTENSIONS),
) -> List[RAGDocument]:
    """
    加载 data/papers 下的所有文本类文档。

    返回：
    List[RAGDocument]
    """
    papers_dir = Path(papers_dir).resolve()

    if not papers_dir.exists():
        raise FileNotFoundError(f"论文目录不存在：{papers_dir}")

    if not papers_dir.is_dir():
        raise NotADirectoryError(f"不是一个目录：{papers_dir}")

    allowed_extensions = {ext.lower() for ext in allowed_extensions}

    documents: List[RAGDocument] = []

    for path in sorted(papers_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        # 跳过隐藏文件或隐藏目录
        if any(part.startswith(".") for part in path.relative_to(papers_dir).parts):
            continue

        text = read_text_file(path, base_dir=papers_dir)

        if not text:
            continue

        relative_path = path.relative_to(ROOT_DIR).as_posix()
        doc_id = path.relative_to(papers_dir).as_posix()

        document = RAGDocument(
            doc_id=doc_id,
            source=relative_path,
            path=str(path),
            text=text,
            num_chars=len(text),
            num_lines=text.count("\n") + 1,
        )

        documents.append(document)

    return documents


def _find_good_split(text: str, start: int, target_end: int, min_end: int) -> int:
    """
    尽量在自然边界处切分。

    优先级：
    - 段落边界
    - 换行
    - 中文句号、问号、感叹号
    - 英文句号、问号、感叹号
    - 分号、逗号

    如果找不到合适边界，就直接按 target_end 硬切。
    """
    separators = (
        "\n\n",
        "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "；", "; ",
        "，", ", ",
    )

    best_pos = -1
    best_len = 0

    for sep in separators:
        pos = text.rfind(sep, min_end, target_end)

        if pos > best_pos:
            best_pos = pos
            best_len = len(sep)

    if best_pos != -1:
        return best_pos + best_len

    return target_end


def _trim_span(text: str, start: int, end: int) -> Tuple[int, int, str]:
    """
    对 chunk 首尾空白做清理，同时修正 start_char / end_char。
    """
    raw = text[start:end]

    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw.rstrip())

    new_start = start + left_trim
    new_end = start + right_trim

    if new_start >= new_end:
        return new_start, new_end, ""

    return new_start, new_end, text[new_start:new_end]


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Tuple[int, int, str]]:
    """
    把一段文本切成多个 chunk。

    参数：
    - chunk_size：每个 chunk 的目标字符数
    - chunk_overlap：相邻 chunk 之间的重叠字符数

    overlap 的作用：
    防止关键信息刚好被切在两个 chunk 边界处。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    text = normalize_text(text)

    if not text:
        return []

    chunks: List[Tuple[int, int, str]] = []
    text_len = len(text)
    start = 0

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end < text_len:
            min_end = min(start + int(chunk_size * 0.6), target_end)
            end = _find_good_split(text, start, target_end, min_end)
        else:
            end = text_len

        chunk_start, chunk_end, chunk_text = _trim_span(text, start, end)

        if chunk_text:
            chunks.append((chunk_start, chunk_end, chunk_text))

        if end >= text_len:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            next_start = end

        start = max(0, next_start)

        # 避免 chunk 开头是一串空白字符
        while start < text_len and text[start].isspace():
            start += 1

    return chunks


def chunk_document(
    document: RAGDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[RAGChunk]:
    """
    把单个 RAGDocument 切成多个 RAGChunk。
    """
    spans = split_text_into_chunks(
        document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[RAGChunk] = []

    for index, (start_char, end_char, chunk_text) in enumerate(spans):
        chunk = RAGChunk(
            chunk_id=f"{document.doc_id}::chunk-{index:04d}",
            doc_id=document.doc_id,
            source=document.source,
            text=chunk_text,
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
            num_chars=len(chunk_text),
        )

        chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: List[RAGDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[RAGChunk]:
    """
    把多个文档统一切块。
    """
    all_chunks: List[RAGChunk] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return all_chunks


QUERY_EXPANSION = {
    # 工具相关
    "工具调用": ["工具执行", "工具选择", "tool calling", "tool execution"],
    "工具执行": ["工具调用", "工具选择", "tool calling", "tool execution"],
    "工具选择": ["工具调用", "工具执行", "tool selection"],

    # 验证/校验相关
    "结果验证": ["验证", "校验", "结果校验", "verification", "verify"],
    "验证": ["结果验证", "校验", "verification", "verify"],
    "校验": ["验证", "结果验证", "verification"],

    # Agent 相关
    "代理系统": ["代理框架", "agent system", "agent", "agent framework"],
    "代理框架": ["代理系统", "agent framework", "agent"],
    "agent": ["代理", "代理系统", "代理框架", "agent system", "agent framework"],

    # 方法/问题相关
    "方法": ["提出", "框架", "method", "approach"],
    "作用": ["用于", "提高", "减少", "解决", "作用", "purpose"],
    "模块": ["任务规划", "工具执行", "内存管理", "记忆管理", "验证", "模块"],

    # RAG 相关，后续文档多了以后会有用
    "rag": ["检索", "增强", "生成", "retrieval", "retrieval augmented generation"],
    "检索": ["rag", "搜索", "召回", "retrieval"],
}


def tokenize_text(text: str) -> List[str]:
    """
    一个不依赖第三方库的简单 tokenizer。

    支持：
    - 英文单词，例如 agent / rag / tool
    - 数字，例如 2024 / 3.5
    - 中文连续片段，例如 方法 / 实验 / 数据集

    注意：
    这不是专业分词器，只是最小可用版本。
    后续接 embedding 后，它会被向量检索替代，或者作为混合检索的一部分。
    """
    text = text.lower()

    english_or_number_terms = re.findall(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)?", text)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]+", text)

    tokens: List[str] = []

    tokens.extend(english_or_number_terms)

    for term in chinese_terms:
        if len(term) <= 2:
            tokens.append(term)
        else:
            # 没有 jieba 时，用中文 bigram 做一个最小可用分词。
            # 例如 “模块化代理框架” -> “模块”“块化”“化代”“代理”“理框”“框架”
            for i in range(len(term) - 1):
                tokens.append(term[i : i + 2])

    return [token for token in tokens if token.strip()]


def expand_query_terms(question: str, terms: List[str]) -> List[str]:
    """
    对用户问题做少量同义词扩展。

    目的：
    - 缓解“问题用词”和“文档用词”不一致导致的检索失败
    - 让 keyword retrieval 更适合作为基础 RAG demo

    例如：
    “工具调用”可以扩展出“工具执行”“工具选择”“tool calling”。
    """
    expanded_terms = list(terms)
    question_lower = question.lower()

    for key, aliases in QUERY_EXPANSION.items():
        key_lower = key.lower()

        # 如果原问题中出现 key，就加入它的同义表达。
        if key_lower in question_lower:
            for alias in aliases:
                expanded_terms.extend(tokenize_text(alias))

    # 去重，但保留顺序
    return list(dict.fromkeys(expanded_terms))


def score_chunk(question: str, chunk: RAGChunk) -> Tuple[float, List[str]]:
    """
    对单个 chunk 进行简单关键词打分。

    打分规则：
    - query 里的词出现在 chunk 中，加分
    - 完整 query 子串出现在 chunk 中，额外加分
    - 命中次数越多，分数越高，但做简单截断，避免长 chunk 刷分
    - 对 query 做同义词扩展，缓解问法与原文措辞不一致的问题
    """
    question_norm = question.lower().strip()
    chunk_norm = chunk.text.lower()

    query_terms = tokenize_text(question)
    query_terms = expand_query_terms(question, query_terms)

    if not query_terms:
        return 0.0, []

    chunk_terms = tokenize_text(chunk.text)
    chunk_counter = Counter(chunk_terms)

    score = 0.0
    matched_terms: List[str] = []

    # 完整问题如果直接出现在 chunk 中，给一个较大 bonus。
    if question_norm and question_norm in chunk_norm:
        score += 5.0

    for term in query_terms:
        term_score = 0.0

        # token 精确命中
        if term in chunk_counter:
            term_score += min(chunk_counter[term], 3) * 1.0

        # 子串命中，适合中文短语或英文短语
        if len(term) >= 2 and term in chunk_norm:
            term_score += 2.0

        if term_score > 0:
            score += term_score
            matched_terms.append(term)

    # 去重，但保留顺序
    unique_matched_terms = list(dict.fromkeys(matched_terms))

    # 轻微长度归一化：避免特别长的 chunk 因为包含词多而过度占优
    length_penalty = max(chunk.num_chars / 800, 1.0)
    normalized_score = score / length_penalty

    return normalized_score, unique_matched_terms


def retrieve(
    question: str,
    chunks: List[RAGChunk],
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    Day 3：最小关键词检索。

    输入：
    - question：用户问题
    - chunks：Day 2 切出来的 RAGChunk
    - top_k：返回前几个最相关片段

    输出：
    - List[RetrievalResult]
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    results: List[RetrievalResult] = []

    for chunk in chunks:
        score, matched_terms = score_chunk(question, chunk)

        if score <= 0:
            continue

        results.append(
            RetrievalResult(
                chunk=chunk,
                score=score,
                matched_terms=matched_terms,
            )
        )

    results.sort(
        key=lambda item: (
            item.score,
            -item.chunk.chunk_index,
        ),
        reverse=True,
    )

    return results[:top_k]


RAG_SYSTEM_PROMPT = """
你是一个严谨的论文阅读助手。

你必须遵守以下规则：
1. 只能基于用户提供的【检索上下文】回答。
2. 不要使用你自己的外部知识补充论文内容。
3. 如果检索上下文中没有足够证据，请明确说明“根据当前检索内容无法确定”。
4. 回答要具体、清晰，优先用中文。
5. 每个关键结论句末都要标注片段编号，例如 [1]、[2]。
6. 引用编号只能来自检索上下文中已有的编号，不能编造 [99] 这种不存在的编号。
7. 不要编造数据集、实验结果、指标、论文结论。
""".strip()


def build_rag_context(
    results: List[RetrievalResult],
    max_context_chars: int = 4000,
) -> str:
    """
    把检索结果组装成 LLM 可读的上下文。

    每个 chunk 会被编号：
    [1] source=...
    text=...

    这些编号后续可以被 LLM 用作引用。
    """
    if max_context_chars <= 0:
        raise ValueError("max_context_chars 必须大于 0")

    context_parts: List[str] = []
    used_chars = 0

    for index, item in enumerate(results, start=1):
        chunk = item.chunk
        citation_label = f"[{index}]"

        part = (
            f"{citation_label}\n"
            f"citation_label: {citation_label}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"source: {chunk.source}\n"
            f"span: {chunk.start_char}-{chunk.end_char}\n"
            f"score: {item.score:.2f}\n"
            f"matched_terms: {', '.join(item.matched_terms)}\n"
            f"text:\n{chunk.text}\n"
        )

        if used_chars + len(part) > max_context_chars:
            remaining = max_context_chars - used_chars

            if remaining <= 200:
                break

            part = part[:remaining] + "\n...[上下文因长度限制被截断]"

        context_parts.append(part)
        used_chars += len(part)

        if used_chars >= max_context_chars:
            break

    return "\n\n".join(context_parts)


def build_rag_prompt(question: str, context: str) -> str:
    """
    构造 RAG 问答 prompt。
    """
    return f"""
请基于下面的【检索上下文】回答用户问题。

【用户问题】
{question}

【检索上下文】
{context}

【回答要求】
- 只能使用检索上下文中的信息。
- 如果证据不足，请明确说明无法确定。
- 每个关键结论句末都要标注引用片段编号，例如 [1]、[2]。
- 引用编号必须来自检索上下文，不要编造不存在的编号。
- 用中文回答。
""".strip()


def answer_with_rag(
    question: str,
    chunks: List[RAGChunk],
    top_k: int = 3,
    max_context_chars: int = 4000,
    min_score: float = 1.0,
) -> RAGAnswerResult:
    """
    Day 4：最小 RAG 回答函数。

    流程：
    1. retrieve(question)
    2. 过滤低分结果
    3. build_rag_context()
    4. call_llm()
    5. 返回 RAGAnswerResult
    """
    retrieval_results = retrieve(
        question=question,
        chunks=chunks,
        top_k=top_k,
    )

    retrieval_results = [
        item for item in retrieval_results
        if item.score >= min_score
    ]

    if not retrieval_results:
        answer = (
            "根据当前检索内容无法确定答案。"
            "可能原因是本地文档中没有相关内容，或者当前关键词检索没有命中有效片段。"
        )

        return RAGAnswerResult(
            question=question,
            answer=answer,
            retrieval_results=[],
            context="",
        )

    context = build_rag_context(
        results=retrieval_results,
        max_context_chars=max_context_chars,
    )

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    answer = call_llm(
        prompt=prompt,
        system_prompt=RAG_SYSTEM_PROMPT,
    )

    return RAGAnswerResult(
        question=question,
        answer=answer,
        retrieval_results=retrieval_results,
        context=context,
    )


def extract_citation_indices(answer: str) -> List[int]:
    """
    从回答文本中提取引用编号。

    例如：
    "该方法包含工具执行和验证模块 [1][2]"
    -> [1, 2]
    """
    matches = re.findall(r"\[(\d+)\]", answer)

    indices: List[int] = []

    for item in matches:
        try:
            index = int(item)
        except ValueError:
            continue

        if index <= 0:
            continue

        indices.append(index)

    # 去重，但保留出现顺序
    return list(dict.fromkeys(indices))


def make_excerpt(text: str, max_chars: int = 240) -> str:
    """
    生成引用片段预览。
    """
    one_line = " ".join(text.split())

    if len(one_line) <= max_chars:
        return one_line

    return one_line[:max_chars] + "..."


def build_citation_sources(
    result: RAGAnswerResult,
    only_cited: bool = False,
    excerpt_chars: int = 240,
) -> List[CitationSource]:
    """
    将回答中的 [1]、[2] 映射到真实检索来源。

    only_cited=True：
    只输出回答中实际出现过的引用编号。

    only_cited=False：
    输出所有用于回答的检索片段，同时标记哪些被回答实际引用了。
    """
    cited_indices = set(extract_citation_indices(result.answer))

    sources: List[CitationSource] = []

    for index, item in enumerate(result.retrieval_results, start=1):
        used_in_answer = index in cited_indices

        if only_cited and not used_in_answer:
            continue

        chunk = item.chunk

        sources.append(
            CitationSource(
                label=f"[{index}]",
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                span=f"{chunk.start_char}-{chunk.end_char}",
                score=item.score,
                matched_terms=item.matched_terms,
                excerpt=make_excerpt(chunk.text, max_chars=excerpt_chars),
                used_in_answer=used_in_answer,
            )
        )

    return sources


def format_citation_sources(
    sources: List[CitationSource],
    title: str = "引用来源：",
) -> str:
    """
    把 citation sources 格式化成命令行输出。
    """
    lines = [title]

    if not sources:
        lines.append("无。")
        return "\n".join(lines)

    for source in sources:
        used_mark = "已在回答中引用" if source.used_in_answer else "未被回答显式引用"
        matched = ", ".join(source.matched_terms) if source.matched_terms else "无"

        lines.extend(
            [
                f"{source.label} {used_mark}",
                f"    source: {source.source}",
                f"    chunk_id: {source.chunk_id}",
                f"    span: {source.span}",
                f"    score: {source.score:.2f}",
                f"    matched_terms: {matched}",
                f"    excerpt: {source.excerpt}",
            ]
        )

    return "\n".join(lines)


def format_document_summary(documents: List[RAGDocument]) -> str:
    """
    把文档加载结果格式化成命令行可读摘要。
    """
    if not documents:
        return "未加载到任何文档。请检查 data/papers 下是否存在 .txt / .md 文件。"

    total_chars = sum(doc.num_chars for doc in documents)
    total_lines = sum(doc.num_lines for doc in documents)

    lines = [
        f"已加载 {len(documents)} 个文档。",
        f"总字符数：{total_chars}",
        f"总行数：{total_lines}",
        "",
        "文档列表：",
    ]

    for index, doc in enumerate(documents, start=1):
        lines.append(
            f"{index}. {doc.doc_id} | chars={doc.num_chars} | lines={doc.num_lines} | source={doc.source}"
        )

    return "\n".join(lines)


def format_chunk_summary(chunks: List[RAGChunk], max_items: int = 10) -> str:
    """
    把 chunk 结果格式化成命令行可读摘要。
    """
    if not chunks:
        return "未生成任何 chunk。"

    total_chars = sum(chunk.num_chars for chunk in chunks)

    lines = [
        f"已生成 {len(chunks)} 个 chunks。",
        f"chunk 总字符数：{total_chars}",
        "",
        f"前 {min(max_items, len(chunks))} 个 chunks：",
    ]

    for index, chunk in enumerate(chunks[:max_items], start=1):
        lines.append(
            (
                f"{index}. {chunk.chunk_id} | "
                f"chars={chunk.num_chars} | "
                f"span={chunk.start_char}-{chunk.end_char} | "
                f"source={chunk.source}"
            )
        )

    if len(chunks) > max_items:
        lines.append(f"... 还有 {len(chunks) - max_items} 个 chunks 未显示。")

    return "\n".join(lines)


def format_retrieval_results(results: List[RetrievalResult]) -> str:
    """
    把检索结果格式化成命令行输出。
    """
    if not results:
        return "没有检索到相关 chunk。"

    lines = [
        f"检索到 {len(results)} 个相关 chunks：",
    ]

    for index, item in enumerate(results, start=1):
        chunk = item.chunk
        matched = ", ".join(item.matched_terms) if item.matched_terms else "无"

        preview = chunk.text[:500]
        if len(chunk.text) > 500:
            preview += "..."

        lines.extend(
            [
                "",
                f"[{index}] score={item.score:.2f}",
                f"chunk_id: {chunk.chunk_id}",
                f"source: {chunk.source}",
                f"span: {chunk.start_char}-{chunk.end_char}",
                f"matched_terms: {matched}",
                "text:",
                preview,
            ]
        )

    return "\n".join(lines)


def format_rag_answer(
    result: RAGAnswerResult,
    show_context: bool = False,
    only_cited_sources: bool = False,
    excerpt_chars: int = 240,
) -> str:
    """
    把 RAG 回答结果格式化成命令行输出。

    Day 5 新增：
    - 提取回答中的 [1] [2]
    - 映射到真实 source / chunk_id / span
    - 输出引用来源
    """
    lines = [
        "RAG 回答：",
        result.answer,
        "",
    ]

    citation_sources = build_citation_sources(
        result=result,
        only_cited=only_cited_sources,
        excerpt_chars=excerpt_chars,
    )

    lines.append(format_citation_sources(citation_sources))

    cited_indices = extract_citation_indices(result.answer)

    if result.retrieval_results and not cited_indices:
        lines.extend(
            [
                "",
                "提示：模型回答中没有显式引用 [1]、[2]。上面仍列出了本次用于回答的检索片段。",
            ]
        )

    if show_context and result.context:
        lines.extend(
            [
                "",
                "传给 LLM 的检索上下文：",
                result.context,
            ]
        )

    return "\n".join(lines)