import json
import re
from typing import Any, Dict, List

from llm import call_llm
from tools import ToolResult, read_file, safe_calculator, search_paper


TOOL_SYSTEM_PROMPT = """
你是一个最小工具调用 Agent。

你只能输出严格 JSON，不要输出解释文字、Markdown 或代码块。

你可以选择两种动作：

1. 调用工具
2. 给出最终回答

可用工具如下：

工具一：calculator
作用：执行安全数学计算。
参数格式：
{
  "expression": "1 + 2 * 3"
}

工具二：search_paper
作用：搜索 data/papers 目录下的论文摘要 txt 文件。
参数格式：
{
  "keyword": "RAG"
}

工具三：read_file
作用：读取 data/papers 目录下指定 txt 文件。
参数格式：
{
  "file_path": "data/papers/demo.txt"
}

如果需要调用工具，必须输出：
{
  "action": "tool",
  "tool_name": "calculator 或 search_paper 或 read_file",
  "arguments": {
    "参数名": "参数值"
  }
}

如果已经可以回答，必须输出：
{
  "action": "final",
  "answer": "最终回答"
}

注意：
- 一次只能调用一个工具。
- 不要编造工具结果。
- 如果工具结果不足，请说明信息不足。
- 只能输出 JSON。
"""


FINAL_SYSTEM_PROMPT = """
你是一个严谨的助手。

请只基于工具执行结果回答用户问题。
如果工具结果中没有足够信息，请明确说明信息不足。
不要编造不存在的论文、文件、实验结果或数据。
"""


def extract_json(text: str) -> dict:
    """
    从模型输出中提取 JSON。

    兼容三种情况：
    1. 模型直接输出 JSON
    2. 模型输出 ```json ... ```
    3. 模型前后夹杂少量文字
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

    raise ValueError(f"无法从模型输出中解析 JSON：{text}")


def dispatch_tool(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    """
    根据模型给出的 tool_name 和 arguments 调用对应工具。
    """
    if tool_name == "calculator":
        expression = arguments.get("expression", "")
        return safe_calculator(expression)

    if tool_name == "search_paper":
        keyword = arguments.get("keyword", "")
        return search_paper(keyword)

    if tool_name == "read_file":
        file_path = arguments.get("file_path", "")
        return read_file(file_path)

    return ToolResult(
        ok=False,
        error=f"未知工具：{tool_name}",
    )


def shorten_text(text: str, max_length: int = 2000) -> str:
    """
    防止工具结果太长，导致后续 prompt 过大。
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + "\n\n...[内容过长，已截断]"


def decide_next_action(question: str, scratchpad: List[dict]) -> dict:
    """
    让 LLM 根据用户问题和已有工具结果，决定下一步动作。
    """
    prompt = f"""
用户问题：
{question}

已有执行过程：
{json.dumps(scratchpad, ensure_ascii=False, indent=2)}

请判断下一步应该调用工具，还是给出最终回答。

只能输出 JSON。
"""

    raw_output = call_llm(
        prompt,
        system_prompt=TOOL_SYSTEM_PROMPT,
    )

    return extract_json(raw_output)

def run_agent_with_trace(question: str, max_steps: int = 3, verbose: bool = False) -> dict:
    """
    带 trace 的最小 Agent Loop。

    返回：
    - question: 用户问题
    - answer: 最终回答
    - scratchpad: 每一步工具调用记录
    """
    scratchpad: List[dict] = []

    for step in range(1, max_steps + 1):
        decision = decide_next_action(question, scratchpad)
        action = decision.get("action")

        if verbose:
            print(f"[Step {step}] LLM 决策：")
            print(json.dumps(decision, ensure_ascii=False, indent=2))

        if action == "final":
            return {
                "question": question,
                "answer": decision.get("answer", ""),
                "scratchpad": scratchpad,
            }

        if action == "tool":
            tool_name = decision.get("tool_name", "")
            arguments = decision.get("arguments", {})

            if not isinstance(arguments, dict):
                arguments = {}

            tool_result = dispatch_tool(tool_name, arguments)

            result_dict = tool_result.to_dict()

            if result_dict.get("result"):
                result_dict["result"] = shorten_text(result_dict["result"])

            trace_item = {
                "step": step,
                "decision": decision,
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_result": result_dict,
            }

            scratchpad.append(trace_item)

            if verbose:
                print(f"[Step {step}] 调用工具：{tool_name}")
                print(f"[Step {step}] 工具参数：")
                print(json.dumps(arguments, ensure_ascii=False, indent=2))
                print(f"[Step {step}] 工具结果：")
                print(json.dumps(result_dict, ensure_ascii=False, indent=2))
                print()

            continue

        scratchpad.append(
            {
                "step": step,
                "error": f"未知 action：{action}",
                "raw_decision": decision,
            }
        )

    final_prompt = f"""
用户问题：
{question}

工具执行过程：
{json.dumps(scratchpad, ensure_ascii=False, indent=2)}

已经达到最大工具调用轮数。
请基于已有工具结果给出最终回答。
如果信息不足，请明确说明。
"""

    answer = call_llm(
        final_prompt,
        system_prompt=FINAL_SYSTEM_PROMPT,
    )

    return {
        "question": question,
        "answer": answer,
        "scratchpad": scratchpad,
    }
def run_agent(question: str, max_steps: int = 3) -> str:
    """
    兼容旧用法：只返回最终答案。
    """
    result = run_agent_with_trace(question, max_steps=max_steps, verbose=False)
    return result["answer"]


def main() -> None:
    test_questions = [
        "计算 24 / 30 * 100 的结果。",
        "本地有没有和 RAG 相关的论文？",
        "读取 data/papers/agent_test.txt，并总结它的方法。",
    ]

    for question in test_questions:
        print("=" * 80)
        print(f"用户问题：{question}")
        print()

        try:
            result = run_agent_with_trace(question, verbose=True)
            answer = result["answer"]
            print("Agent 回答：")
            print(answer)

        except Exception as e:
            print("Agent 运行失败：")
            print(e)

        print()


if __name__ == "__main__":
    main()