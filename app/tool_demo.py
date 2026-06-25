from tools import safe_calculator


def main() -> None:
    test_expressions = [
        "1 + 2 * 3",
        "(10 - 3) / 7",
        "2 ** 5",
        "10 // 3",
        "10 % 3",
        "1 / 0",
        "__import__('os').system('dir')",
    ]

    for expression in test_expressions:
        result = safe_calculator(expression)

        if result.ok:
            print(f"[成功] {expression} = {result.result}")
        else:
            print(f"[失败] {expression}，原因：{result.error}")


if __name__ == "__main__":
    main()