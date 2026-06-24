import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas import PaperSummary
from llm import call_llm


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "outputs" / "logs"


def save_log(name: str, content: str) -> None:
    """
    保存运行日志，方便以后排查 prompt、模型输出和报错。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{name}.txt"

    log_path.write_text(content, encoding="utf-8")


def resolve_path(path_str: str) -> Path:
    """
    将命令行传入的路径转换成绝对路径。

    如果传入的是绝对路径，例如：
    D:\\xxx\\demo.txt

    就直接使用。

    如果传入的是相对路径，例如：
    data\\papers\\demo.txt

    就默认从项目根目录 ROOT_DIR 开始寻找。
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def build_output_path(input_path: Path, output_arg: Optional[str]) -> Path:
    """
    如果用户指定了 --output，就使用用户指定的输出路径。
    如果没有指定，就自动生成：
    outputs/输入文件名_summary.json
    """
    if output_arg:
        output_path = Path(output_arg)

        if output_path.is_absolute():
            return output_path

        return ROOT_DIR / output_path

    return ROOT_DIR / "outputs" / f"{input_path.stem}_summary.json"


def build_batch_output_path(input_path: Path, output_dir: Path) -> Path:
    """
    批量模式下，根据输入 txt 文件自动生成输出 json 路径。

    例如：
    data/papers/demo.txt
    -> outputs/demo_summary.json
    """
    return output_dir / f"{input_path.stem}_summary.json"


def extract_json(text: str) -> dict:
    """
    尝试从模型输出中提取 JSON。
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("无法从模型输出中解析 JSON。")


def summarize_paper_abstract(abstract: str) -> PaperSummary:
    prompt = f"""
请阅读下面的论文摘要，并输出严格 JSON。

要求：
1. 只输出 JSON，不要输出解释文字。
2. 所有字段内容使用中文。
3. JSON 字段必须包含：
   - problem: 研究问题
   - method: 方法概述
   - datasets: 使用的数据集，如果摘要未提到则输出空列表
   - contributions: 主要贡献列表
   - limitations: 局限列表，如果摘要未提到则输出空列表
4. 不要编造摘要中没有提到的具体实验结果。
5. datasets、contributions、limitations 必须是字符串列表。

论文摘要：
{abstract}

输出 JSON 格式如下：
{{
  "problem": "...",
  "method": "...",
  "datasets": ["..."],
  "contributions": ["..."],
  "limitations": ["..."]
}}
"""

    save_log("prompt", prompt)

    raw_output = call_llm(prompt)
    save_log("raw_output", raw_output)

    try:
        data = extract_json(raw_output)
    except Exception as e:
        save_log("json_parse_error", str(e))
        raise ValueError(f"无法解析模型输出为 JSON：{e}")

    try:
        return PaperSummary(**data)
    except ValidationError as e:
        save_log("validation_error", str(e))
        raise ValueError(f"模型输出格式不符合 PaperSummary：{e}")


def process_one_file(input_path: Path, output_path: Path) -> None:
    """
    处理单个论文摘要文件：
    1. 读取 txt
    2. 调用 LLM 生成结构化总结
    3. 保存 json
    """
    input_path = resolve_path(str(input_path))
    output_path = resolve_path(str(output_path))

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    abstract = input_path.read_text(encoding="utf-8").strip()

    if not abstract:
        raise ValueError(f"输入文件是空文件，请先放入一段论文摘要：{input_path}")

    summary = summarize_paper_abstract(abstract)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_batch(input_dir: Path, output_dir: Path) -> None:
    """
    批量处理 input_dir 目录下所有 .txt 文件。
    """
    input_dir = resolve_path(str(input_dir))
    output_dir = resolve_path(str(output_dir))

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input-dir 不是目录：{input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))

    print(f"开始批量处理，共发现 {len(txt_files)} 个 txt 文件。")

    success_count = 0
    fail_count = 0

    for txt_file in txt_files:
        output_path = build_batch_output_path(txt_file, output_dir)

        try:
            process_one_file(txt_file, output_path)
            print(f"[成功] {txt_file.name} -> {output_path.name}")
            success_count += 1

        except Exception as e:
            print(f"[失败] {txt_file.name}，原因：{e}")
            save_log(
                f"batch_error_{txt_file.stem}",
                f"文件：{txt_file}\n输出：{output_path}\n错误：{repr(e)}",
            )
            fail_count += 1

    print()
    print("批量处理完成。")
    print(f"成功：{success_count}")
    print(f"失败：{fail_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="论文摘要结构化总结工具"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="data/papers/demo.txt",
        help="输入论文摘要 txt 文件路径，默认 data/papers/demo.txt",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 JSON 文件路径。默认输出到 outputs/输入文件名_summary.json",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理 input-dir 下的所有 txt 文件。",
    )

    parser.add_argument(
        "--input-dir",
        default="data/papers",
        help="批量模式下的输入目录，默认 data/papers。",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="批量模式下的输出目录，默认 outputs。",
    )

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        if args.batch:
            process_batch(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
            )
            return

        input_path = resolve_path(args.input)
        output_path = build_output_path(input_path, args.output)

        process_one_file(input_path, output_path)

        print("已生成结构化总结：")
        print(output_path)

    except Exception as e:
        save_log("fatal_error", str(e))
        raise


if __name__ == "__main__":
    main()