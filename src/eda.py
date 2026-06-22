"""
CIC-IDS2017 探索性数据分析（EDA）模块
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 确保中文字体显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_label_distribution(df: pd.DataFrame, save: bool = True):
    """绘制标签分布（二分类 + 原始）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 二分类分布
    binary_counts = df["Label_Binary"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(binary_counts.index, binary_counts.values, color=colors, alpha=0.8)
    axes[0].set_title("Binary Label Distribution (BENIGN vs Malicious)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(binary_counts.values):
        axes[0].text(i, v + 10000, f"{v:,}", ha="center")

    # 原始标签分布（Top 15）
    orig_counts = df["Label"].value_counts()
    axes[1].barh(orig_counts.index[:15], orig_counts.values[:15], color="steelblue", alpha=0.8)
    axes[1].set_title("Top 15 Original Labels")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(orig_counts.values[:15]):
        axes[1].text(v + 5000, i, f"{v:,}", va="center")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "label_distribution.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, save: bool = True):
    """绘制正负样本比例"""
    binary_counts = df["Label_Binary"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        binary_counts.values,
        labels=binary_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        explode=(0, 0.05),
    )
    ax.set_title("Class Balance (BENIGN vs Malicious)")
    if save:
        path = RESULTS_DIR / "class_balance.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)


def plot_day_distribution(df: pd.DataFrame, save: bool = True):
    """按天统计流量和攻击分布"""
    day_order = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    day_counts = df.groupby("Day")["Label_Binary"].value_counts().unstack(fill_value=0)
    day_counts = day_counts.reindex(day_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    day_counts.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"], alpha=0.8)
    ax.set_title("Daily Traffic Distribution: BENIGN vs Malicious")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend(title="Label")
    ax.set_xticklabels([d.capitalize() for d in day_order], rotation=0)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "day_distribution.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)


def plot_top_features(df: pd.DataFrame, top_n: int = 20, save: bool = True):
    """
    使用 ANOVA F-value 快速筛选 Top-N 特征
    （快速初步筛选，非最终特征重要性）
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    # 排除非特征列
    label_cols = {"Label", "Label_Binary", "Day"}
    feature_cols = [c for c in df.columns if c not in label_cols]

    # 取子集加速
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = (df["Label_Binary"] == "Malicious").astype(int).values

    # 采样 50000 条以加速
    if len(df) > 50000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), 50000, replace=False)
        X, y = X[idx], y[idx]

    selector = SelectKBest(f_classif, k=top_n)
    selector.fit(X, y)
    scores = selector.scores_

    # 排序取 Top-N
    top_idx = np.argsort(scores)[-top_n:][::-1]
    top_features = [feature_cols[i] for i in top_idx]
    top_scores = scores[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_scores, color="coral", alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(f"Top {top_n} Features by ANOVA F-Score")
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "top_features_anova.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)

    return list(zip(top_features, top_scores))


def plot_correlation_heatmap(df: pd.DataFrame, sample_size: int = 30000, save: bool = True):
    """绘制特征相关性热力图"""
    label_cols = {"Label", "Label_Binary", "Day"}
    feature_cols = [c for c in df.columns if c not in label_cols]

    # 采样加速
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    corr = df_sample[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "correlation_heatmap.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)


def generate_eda_report(df: pd.DataFrame) -> dict:
    """生成 EDA 报告摘要"""
    report = {
        "total_records": len(df),
        "num_features": len([c for c in df.columns if c not in {"Label", "Label_Binary", "Day"}]),
        "benign_count": int((df["Label_Binary"] == "BENIGN").sum()),
        "malicious_count": int((df["Label_Binary"] == "Malicious").sum()),
        "attack_ratio": round((df["Label_Binary"] == "Malicious").mean() * 100, 2),
        "num_attack_types": df["Label"].nunique() - 1,  # 减去 BENIGN
        "nan_count": int(df.isnull().sum().sum()),
        "inf_count": int((df.select_dtypes(include=[np.number]) == float("inf")).sum().sum()
                         + (df.select_dtypes(include=[np.number]) == float("-inf")).sum().sum()),
    }

    print("\n" + "=" * 50)
    print("EDA 报告摘要")
    print("=" * 50)
    print(f"总记录数:        {report['total_records']:,}")
    print(f"特征维度:        {report['num_features']}")
    print(f"正常流量 (BENIGN):  {report['benign_count']:,}")
    print(f"恶意流量:        {report['malicious_count']:,}")
    print(f"恶意流量占比:    {report['attack_ratio']}%")
    print(f"攻击类型数:      {report['num_attack_types']}")
    print(f"缺失值:          {report['nan_count']:,}")
    print(f"无穷值:          {report['inf_count']:,}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    from data_loader import load_all_data

    df = load_all_data()

    report = generate_eda_report(df)
    plot_label_distribution(df)
    plot_class_balance(df)
    plot_day_distribution(df)
    plot_top_features(df, top_n=20)
    plot_correlation_heatmap(df)

    print("\nEDA 完成！所有图表已保存到 results/ 目录")
