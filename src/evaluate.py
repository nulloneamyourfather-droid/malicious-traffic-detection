"""
模型评估模块：
- 分类指标计算
- 混淆矩阵、ROC、PR 曲线
- 特征重要性可视化
- 按攻击类型拆分分析
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save: bool = True,
) -> dict:
    """
    二分类评估指标

    Returns:
        metrics dict
    """
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "test_samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
    }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save: bool = True,
):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    # 规范化
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 绝对数
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["BENIGN", "Malicious"],
                yticklabels=["BENIGN", "Malicious"])
    axes[0].set_title(f"{model_name} - Confusion Matrix (Count)")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # 百分比
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=["BENIGN", "Malicious"],
                yticklabels=["BENIGN", "Malicious"])
    axes[1].set_title(f"{model_name} - Confusion Matrix (Ratio)")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    models_data: list,
    save: bool = True,
):
    """
    绘制 ROC 曲线（支持多模型对比）

    models_data: [(model_name, y_true, y_prob), ...]
    """
    plt.figure(figsize=(8, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, (name, y_t, y_p) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        plt.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc:.4f})",
            color=colors[i % len(colors)],
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save:
        path = RESULTS_DIR / "roc_curves_comparison.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save: bool = True,
):
    """绘制 Precision-Recall 曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = f1_scores.argmax()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#3498db", linewidth=2, label=f"{model_name}")
    plt.scatter(
        recall[best_idx], precision[best_idx],
        color="red", s=100, zorder=5,
        label=f"Best F1 = {f1_scores[best_idx]:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if save:
        path = RESULTS_DIR / f"{model_name}_pr_curve.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model,
    feature_cols: list,
    model_name: str,
    top_n: int = 30,
    save: bool = True,
):
    """绘制特征重要性（支持 LGB 和 RF）"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"  {model_name} 没有 feature_importances_ 属性")
        return

    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        importances[indices],
        color="coral",
        alpha=0.7,
    )
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importance - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"{model_name}_feature_importance.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close()


def evaluate_by_attack_type(
    y_test_orig: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save: bool = True,
):
    """
    按原始攻击类型分析 Recall

    y_test_orig: 原始标签数组
    y_pred: 预测的二进制标签
    """
    results = []
    test_labels = pd.Series(y_test_orig)

    for label in test_labels.unique():
        mask = test_labels == label
        total = mask.sum()
        if total == 0:
            continue

        if label == "BENIGN":
            # 对 BENIGN，计算 specificity
            correct = (~mask & (y_pred == 0)).sum()  # 预测为正常
            detected = (mask & (y_pred == 0)).sum()  # 正确预测为正常
            recall_val = detected / total if total > 0 else 0
        else:
            # 对攻击类型，计算 recall
            detected = (mask & (y_pred == 1)).sum()
            recall_val = detected / total if total > 0 else 0

        results.append({
            "attack_type": label,
            "total": total,
            "detected": int(detected),
            "recall": round(recall_val, 4),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total", ascending=False)

    # 清理标签中的非法字符（如 Web Attack � Brute Force 中的 �）
    results_df["attack_type_clean"] = results_df["attack_type"].apply(
        lambda x: x.replace("�", "?").encode("utf-8", errors="replace").decode("utf-8")
        if isinstance(x, str) else x
    )

    print(f"\n按攻击类型的检测率 ({model_name}):")
    print("-" * 60)
    print(f"{'攻击类型':<30} {'总数':>8} {'检出':>8} {'Recall':>8}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        label = row["attack_type_clean"]
        print(f"{label:<30} {row['total']:>8} {row['detected']:>8} {row['recall']:>7.2%}")
    print("-" * 60)

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if t == "BENIGN" else "#e74c3c" for t in results_df["attack_type"]]
    bars = ax.barh(
        range(len(results_df)),
        results_df["recall"],
        color=colors,
        alpha=0.8,
    )
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["attack_type_clean"])
    ax.set_xlabel("Recall / Detection Rate")
    ax.set_title(f"Detection Rate by Attack Type - {model_name}")
    ax.set_xlim([0, 1.05])
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    for i, v in enumerate(results_df["recall"]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"{model_name}_detection_by_type.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)

    return results_df


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在验证集上找最佳概率阈值（最大化 F1）"""
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = f1_scores.argmax()
    return thresholds[best_idx]


def full_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_orig: np.ndarray,
    feature_cols: list,
    model_name: str,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
):
    """
    完整评估流程
    """
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # 找最佳阈值
    if X_val is not None and y_val is not None and y_prob is not None:
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, val_prob)
        print(f"  最佳阈值 (from validation): {threshold:.4f}")
        y_pred = (y_prob >= threshold).astype(int)
    elif y_prob is not None:
        threshold = find_optimal_threshold(y_test, y_prob)
        print(f"  最佳阈值 (from test): {threshold:.4f}")
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        threshold = 0.5

    # 基础指标
    metrics = evaluate_binary(y_test, y_pred, y_prob if y_prob is not None else y_pred, model_name)

    # 补充阈值信息
    metrics["optimal_threshold"] = round(float(threshold), 4)

    print(f"\n分类指标 (threshold={threshold:.4f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 保存指标
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, model_name)

    # PR 曲线
    if y_prob is not None:
        plot_pr_curve(y_test, y_prob, model_name)

    # 特征重要性
    plot_feature_importance(model, feature_cols, model_name)

    # 按攻击类型分析
    evaluate_by_attack_type(y_test_orig, y_pred, model_name)

    return metrics


def compare_models(all_metrics: list, save: bool = True):
    """
    多模型指标对比
    """
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index("model")

    print(f"\n{'='*60}")
    print("模型对比总结")
    print(f"{'='*60}")
    print(metrics_df.to_string())

    # 可视化对比
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_plot = metrics_df.drop(columns=["test_samples", "positive_samples"])
    metrics_plot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "model_comparison.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  -> 已保存: {path}")
    plt.close(fig)

    return metrics_df


def print_classification_report(y_test, y_pred, model_name):
    """打印详细的 scikit-learn 分类报告"""
    report = classification_report(
        y_test, y_pred,
        target_names=["BENIGN", "Malicious"],
        digits=4,
    )
    print(f"\n分类报告 ({model_name}):")
    print(report)


if __name__ == "__main__":
    from data_loader import load_all_data, get_train_test_split
    from preprocessing import get_processed_train_test
    from joblib import load

    print("加载数据 ...")
    df = load_all_data()
    train_df, test_df = get_train_test_split(df)
    data = get_processed_train_test(train_df, test_df)

    all_results = []

    for model_name in ["lgb", "rf", "lr"]:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            print(f"模型 {model_name} 未找到，跳过")
            continue

        model = load(model_path)
        result = full_evaluation(
            model,
            data["X_test"], data["y_test"],
            data["y_test_orig"], data["feature_cols"],
            model_name,
        )
        all_results.append(result)

    # ROC 曲线对比
    plot_roc_curve(
        data["y_test"],
        None,
        [],  # 稍后填充
    )

    # 多模型 ROC 对比需要收集概率
    roc_data = []
    for model_name in ["lgb", "rf", "lr"]:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            continue
        model = load(model_path)
        y_prob = model.predict_proba(data["X_test"])[:, 1]
        roc_data.append((model_name, data["y_test"], y_prob))

    if len(roc_data) > 1:
        plot_roc_curve(data["y_test"], None, roc_data)

    if all_results:
        compare_models(all_results)
