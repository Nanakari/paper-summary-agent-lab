from tools import read_file


def main() -> None:
    test_files = [
        "data/papers/demo.txt",
        "data/papers/agent_test.txt",
        ".env",
        "README.md",
        "data/papers/not_exist.txt",
    ]

    for file_path in test_files:
        print("=" * 60)
        print(f"[读取] {file_path}")

        result = read_file(file_path)

        if result.ok:
            print("[成功]")
            print(result.result[:300])
        else:
            print("[失败]")
            print(result.error)


if __name__ == "__main__":
    main()