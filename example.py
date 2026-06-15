import pandas as pd
from fuzzy_merge import (
    calculate_similarity,
    find_best_match,
    fuzzy_merge,
    fuzzy_match_report,
)


def main():
    print("=" * 70)
    print("模糊匹配合并服务 - 演示示例")
    print("=" * 70)

    print("\n【1】字符串相似度计算示例")
    print("-" * 70)
    test_pairs = [
        ("Apple Inc.", "Apple Inc"),
        ("Microsoft Corporation", "Microsft Corp"),
        ("Google", "Alphabet"),
        ("腾讯科技", "腾讯科技有限公司"),
    ]
    for s1, s2 in test_pairs:
        score = calculate_similarity(s1, s2)
        print(f"'{s1}' vs '{s2}' => 相似度: {score:.4f}")

    print("\n【2】查找最佳匹配示例")
    print("-" * 70)
    candidates = ["Apple Inc.", "Microsoft Corp", "Google LLC", "Amazon.com", "Meta Platforms"]
    target = "Appel Inc"
    best = find_best_match(target, candidates, threshold=0.5)
    print(f"目标字符串: '{target}'")
    print(f"候选列表: {candidates}")
    if best:
        print(f"最佳匹配: '{best[0]}' (相似度: {best[1]:.4f})")
    else:
        print("未找到匹配项")

    top3 = find_best_match(target, candidates, threshold=0.3, top_n=3)
    print(f"\nTop 3 匹配结果:")
    for i, (match, score) in enumerate(top3, 1):
        print(f"  {i}. '{match}' - {score:.4f}")

    print("\n【3】构建示例数据集")
    print("-" * 70)
    df_sales = pd.DataFrame({
        "company_name": [
            "Apple Inc.",
            "Microsoft Corporation",
            "Google LLC",
            "Amazon.com Inc",
            "Meta Platforms",
            "Tesla Motors",
            "Unknown Company",
        ],
        "revenue_2024": [383.3, 211.9, 307.4, 574.8, 134.9, 96.8, 10.0],
    })

    df_stock = pd.DataFrame({
        "company": [
            "Apple Inc",
            "Microsft Corp",
            "Alphabet Inc",
            "Amazon",
            "Meta",
            "Tesla Inc",
            "NVIDIA Corp",
        ],
        "stock_price": [195.2, 420.5, 165.8, 185.3, 485.2, 250.1, 920.4],
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"],
    })

    print("左表 (销售数据):")
    print(df_sales.to_string(index=False))
    print("\n右表 (股票数据):")
    print(df_stock.to_string(index=False))

    print("\n【4】模糊匹配合并示例 (左连接, 阈值=0.6)")
    print("-" * 70)
    merged_df = fuzzy_merge(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        threshold=0.6,
        how="left",
    )
    print(merged_df.to_string(index=False))

    print("\n【5】模糊匹配合并示例 (外连接, 阈值=0.7)")
    print("-" * 70)
    merged_outer = fuzzy_merge(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        threshold=0.7,
        how="outer",
    )
    print(merged_outer.to_string(index=False))

    print("\n【6】匹配报告示例 (显示 Top 5 候选)")
    print("-" * 70)
    report = fuzzy_match_report(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        threshold=0.3,
    )
    print(report.to_string(index=False))

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
