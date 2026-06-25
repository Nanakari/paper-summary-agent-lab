import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("LLM_MIN_CALL_INTERVAL_SECONDS", "5"))

_LAST_CALL_TIME = 0.0

DEFAULT_SYSTEM_PROMPT = "你是一个严谨的论文阅读助手。"


def wait_for_rate_limit() -> None:
    """
    简单的全局限速器。

    保证相邻两次 Gemini API 调用之间至少间隔
    MIN_CALL_INTERVAL_SECONDS 秒。
    """
    global _LAST_CALL_TIME

    now = time.time()
    elapsed = now - _LAST_CALL_TIME

    if elapsed < MIN_CALL_INTERVAL_SECONDS:
        sleep_seconds = MIN_CALL_INTERVAL_SECONDS - elapsed
        print(f"等待 {sleep_seconds:.1f} 秒，避免触发 API 限流...")
        time.sleep(sleep_seconds)

    _LAST_CALL_TIME = time.time()


def extract_retry_delay(error: Exception) -> float:
    """
    从 Gemini 429 错误信息中尝试提取建议等待时间。

    兼容类似：
    - Please retry in 27.003674233s
    - 'retryDelay': '27s'
    """
    message = str(error)

    match = re.search(r"Please retry in ([0-9.]+)s", message)
    if match:
        return float(match.group(1))

    match = re.search(r"'retryDelay': '([0-9.]+)s'", message)
    if match:
        return float(match.group(1))

    return 10.0


def call_llm(
    prompt: str,
    max_retries: int = 3,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    调用 Gemini API，并在失败时自动重试。

    参数：
    - prompt: 用户输入的主要提示词
    - max_retries: 最大重试次数
    - system_prompt: 系统提示词，用于控制模型角色和输出风格
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("未找到 GEMINI_API_KEY，请检查 .env 文件。")

    client = genai.Client(api_key=api_key)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=system_prompt,
            )

            wait_for_rate_limit()

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise ValueError("Gemini 没有返回有效文本。")

            return response.text

        except Exception as e:
            last_error = e
            print(f"第 {attempt} 次调用 Gemini 失败：{e}")

            if attempt < max_retries:
                wait_seconds = extract_retry_delay(e)

                if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    wait_seconds = min(wait_seconds, 5.0)

                print(f"等待 {wait_seconds:.1f} 秒后重试...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini 调用失败，已重试 {max_retries} 次。最后错误：{last_error}"
    )