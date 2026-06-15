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

    print("\n【1】多种相似度算法对比示例")
    print("-" * 70)
    algorithms = ["difflib", "levenshtein", "jaccard", "cosine"]
    test_pairs = [
        ("Apple Inc.", "Apple Inc"),
        ("Microsoft Corporation", "Microsft Corp"),
        ("Amazon.com Inc", "Amazon"),
        ("Google", "Alphabet"),
        ("腾讯科技", "腾讯科技有限公司"),
    ]
    print(f"{'字符串对':<45} {'difflib':>8} {'levenshtein':>11} {'jaccard':>8} {'cosine':>8}")
    print("-" * 90)
    for s1, s2 in test_pairs:
        scores = []
        for algo in algorithms:
            score = calculate_similarity(s1, s2, algorithm=algo)
            scores.append(f"{score:.4f}")
        pair_str = f"'{s1}' vs '{s2}'"
        print(f"{pair_str:<45} {scores[0]:>8} {scores[1]:>11} {scores[2]:>8} {scores[3]:>8}")

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

    print("\n【4】模糊匹配合并示例 (左连接, 默认阈值=0.8)")
    print("-" * 70)
    merged_df = fuzzy_merge(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        how="left",
    )
    print(merged_df.to_string(index=False))

    print("\n【5】阈值对比：低阈值=0.6 (可能导致误匹配)")
    print("-" * 70)
    merged_low = fuzzy_merge(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        threshold=0.6,
        how="left",
    )
    print(merged_low.to_string(index=False))

    print("\n【6】模糊匹配合并示例 (外连接, 默认阈值=0.8)")
    print("-" * 70)
    merged_outer = fuzzy_merge(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
        how="outer",
    )
    print(merged_outer.to_string(index=False))

    print("\n【7】匹配报告示例 (显示 Top 5 候选, 默认阈值=0.8)")
    print("-" * 70)
    report = fuzzy_match_report(
        df_sales,
        df_stock,
        left_on="company_name",
        right_on="company",
    )
    print(report.to_string(index=False))

    print("\n【8】多种算法模糊合并效果对比 (阈值=0.6)")
    print("-" * 70)
    for algo in ["difflib", "levenshtein", "jaccard", "cosine"]:
        print(f"\n--- 使用 {algo} 算法 ---")
        merged_algo = fuzzy_merge(
            df_sales,
            df_stock,
            left_on="company_name",
            right_on="company",
            threshold=0.6,
            how="left",
            algorithm=algo,
        )
        print(merged_algo[["company_name", "company", "similarity", "algorithm"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
