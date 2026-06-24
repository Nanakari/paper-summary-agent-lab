import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """
    调用 Gemini API，并在失败时自动重试。
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("未找到 GEMINI_API_KEY，请检查 .env 文件。")

    client = genai.Client(api_key=api_key)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )

            if not response.text:
                raise ValueError("Gemini 没有返回有效文本。")

            return response.text

        except Exception as e:
            last_error = e
            print(f"第 {attempt} 次调用 Gemini 失败：{e}")

            if attempt < max_retries:
                time.sleep(2)

    raise RuntimeError(
        f"Gemini 调用失败，已重试 {max_retries} 次。最后错误：{last_error}"
    )
