import json
from pathlib import Path
import ast
import operator
import re
from dataclasses import dataclass
from typing import Optional, Union


Number = Union[int, float]


@dataclass
class ToolResult:
    """
    工具执行结果。

    ok=True 表示工具执行成功。
    ok=False 表示工具执行失败，error 里保存失败原因。
    """
    ok: bool
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
        }


MAX_EXPR_LENGTH = 100
MAX_ABS_VALUE = 1_000_000_000
MAX_POWER = 10


ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_number(value: Number) -> Number:
    """
    检查数值是否在安全范围内，避免过大的计算。
    """
    if isinstance(value, bool):
        raise ValueError("不支持布尔值。")

    if not isinstance(value, (int, float)):
        raise ValueError("只支持整数和小数。")

    if abs(value) > MAX_ABS_VALUE:
        raise ValueError(f"计算结果过大，超过限制：{MAX_ABS_VALUE}")

    return value


def _eval_node(node: ast.AST) -> Number:
    """
    递归计算 AST 节点，只允许安全的数学表达式。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        return _check_number(node.value)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        op_type = type(node.op)

        if op_type not in ALLOWED_BIN_OPS:
            raise ValueError("不支持的二元运算。")

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("不能除以 0。")

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise ValueError(f"指数过大，最大只允许 {MAX_POWER}。")

        result = ALLOWED_BIN_OPS[op_type](left, right)
        return _check_number(result)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("不支持的一元运算。")

        operand = _eval_node(node.operand)
        result = ALLOWED_UNARY_OPS[op_type](operand)
        return _check_number(result)

    raise ValueError(f"不支持的表达式类型：{type(node).__name__}")


def safe_calculator(expression: str) -> ToolResult:
    """
    安全计算器工具。

    支持：
    - 加减乘除：+ - * /
    - 整除：//
    - 取余：%
    - 幂运算：**
    - 括号
    - 整数和小数

    不支持：
    - 函数调用
    - 变量
    - 文件读取
    - 系统命令
    - Python 代码执行
    """
    expression = expression.strip()

    if not expression:
        return ToolResult(ok=False, error="表达式为空。")

    if len(expression) > MAX_EXPR_LENGTH:
        return ToolResult(
            ok=False,
            error=f"表达式过长，最大长度为 {MAX_EXPR_LENGTH}。",
        )

    if not re.fullmatch(r"[0-9+\-*/().% \t]+", expression):
        return ToolResult(
            ok=False,
            error="表达式包含非法字符，只允许数字、运算符和括号。",
        )

    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        return ToolResult(ok=True, result=str(value))

    except Exception as e:
        return ToolResult(ok=False, error=str(e))
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PAPERS_DIR = ROOT_DIR / "data" / "papers"


def search_paper(keyword: str, papers_dir: Optional[str] = None) -> ToolResult:
    """
    搜索本地 data/papers 目录下的 txt 论文摘要文件。

    支持搜索：
    - 文件名
    - 文件内容

    返回匹配到的文件名和内容片段。
    """
    keyword = keyword.strip()

    if not keyword:
        return ToolResult(ok=False, error="搜索关键词不能为空。")

    if len(keyword) > 100:
        return ToolResult(ok=False, error="搜索关键词过长，最大长度为 100。")

    if papers_dir is None:
        search_dir = DEFAULT_PAPERS_DIR
    else:
        search_dir = Path(papers_dir)

        if not search_dir.is_absolute():
            search_dir = ROOT_DIR / search_dir

    if not search_dir.exists():
        return ToolResult(ok=False, error=f"论文目录不存在：{search_dir}")

    if not search_dir.is_dir():
        return ToolResult(ok=False, error=f"指定路径不是目录：{search_dir}")

    results = []
    keyword_lower = keyword.lower()

    for txt_file in sorted(search_dir.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            results.append(
                {
                    "file": txt_file.name,
                    "matched": False,
                    "error": f"读取失败：{e}",
                }
            )
            continue

        filename_lower = txt_file.name.lower()
        content_lower = content.lower()

        matched_in_filename = keyword_lower in filename_lower
        matched_in_content = keyword_lower in content_lower

        if matched_in_filename or matched_in_content:
            match_type = []

            if matched_in_filename:
                match_type.append("filename")

            if matched_in_content:
                match_type.append("content")

            index = content_lower.find(keyword_lower)

            if index >= 0:
                start = max(0, index - 60)
                end = min(len(content), index + len(keyword) + 120)
                snippet = content[start:end].replace("\n", " ")
            else:
                snippet = content[:180].replace("\n", " ")

            results.append(
                {
                    "file": txt_file.name,
                    "matched": True,
                    "match_type": match_type,
                    "snippet": snippet,
                }
            )

    if not results:
        return ToolResult(
            ok=True,
            result=json.dumps(
                {
                    "query": keyword,
                    "matches": [],
                    "message": "没有找到匹配的论文摘要文件。",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    return ToolResult(
        ok=True,
        result=json.dumps(
            {
                "query": keyword,
                "matches": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
ALLOWED_READ_DIRS = [
    ROOT_DIR / "data" / "papers",
]


def is_path_allowed(path: Path) -> bool:
    """
    判断目标路径是否在允许读取的目录下。

    这样可以防止工具读取 .env、系统文件、SSH key 等敏感文件。
    """
    path = path.resolve()

    for allowed_dir in ALLOWED_READ_DIRS:
        allowed_dir = allowed_dir.resolve()

        try:
            path.relative_to(allowed_dir)
            return True
        except ValueError:
            continue

    return False


def read_file(file_path: str) -> ToolResult:
    """
    安全读取本地 txt 文件。

    当前只允许读取 data/papers 目录下的 .txt 文件。
    """
    file_path = file_path.strip()

    if not file_path:
        return ToolResult(ok=False, error="文件路径不能为空。")

    path = Path(file_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    path = path.resolve()

    if not is_path_allowed(path):
        return ToolResult(
            ok=False,
            error=f"禁止读取白名单目录外的文件：{path}",
        )

    if not path.exists():
        return ToolResult(ok=False, error=f"文件不存在：{path}")

    if not path.is_file():
        return ToolResult(ok=False, error=f"路径不是文件：{path}")

    if path.suffix.lower() != ".txt":
        return ToolResult(ok=False, error="当前只支持读取 .txt 文件。")

    try:
        content = path.read_text(encoding="utf-8")

        if not content.strip():
            return ToolResult(ok=False, error=f"文件为空：{path}")

        return ToolResult(ok=True, result=content)

    except Exception as e:
        return ToolResult(ok=False, error=f"读取文件失败：{e}")