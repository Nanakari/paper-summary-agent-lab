from tools import search_paper


def main() -> None:
    keywords = [
        "RAG",
        "agent",
        "工具",
        "不存在的关键词",
    ]

    for keyword in keywords:
        print("=" * 60)
        print(f"[查询] {keyword}")

        result = search_paper(keyword)

        if result.ok:
            print("[成功]")
            print(result.result)
        else:
            print("[失败]")
            print(result.error)


if __name__ == "__main__":
    main()