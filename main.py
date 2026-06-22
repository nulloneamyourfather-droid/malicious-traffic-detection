#!/usr/bin/env python
"""
CIC-IDS2017 恶意流量检测系统 — 主入口

使用方法:
    # 完整流程：训练 + 评估（默认随机分层采样，所有模型）
    python main.py --mode train

    # 用时间划分（模拟零日攻击检测）
    python main.py --mode train --split time

    # 只训练 LightGBM（最快）
    python main.py --mode train --model lgb

    # 仅 EDA 分析
    python main.py --mode eda

    # 对新数据进行预测
    python main.py --mode predict --input data/new_traffic.csv --model lgb
"""
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def cmd_train(args):
    """完整的训练 + 评估流程"""
    from data_loader import (
        load_all_data,
        get_train_test_split,
        get_random_stratified_split,
    )
    from preprocessing import get_processed_train_test
    from train import train_model
    from evaluate import full_evaluation, plot_roc_curve, compare_models
    from joblib import load

    print("=" * 60)
    print("  CIC-IDS2017 恶意流量检测模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据 ...")
    df = load_all_data()

    # 2. 数据划分
    if args.split == "time":
        print("\n[2/5] 按时间划分训练/测试集 ...")
        train_df, test_df = get_train_test_split(df)
    else:
        print("\n[2/5] 随机分层采样划分训练/测试集 ...")
        train_df, test_df = get_random_stratified_split(df, test_size=args.test_size)

    # 3. 特征工程
    print("\n[3/5] 特征工程 ...")
    data = get_processed_train_test(train_df, test_df)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    y_test_orig = data["y_test_orig"]
    feature_cols = data["feature_cols"]

    # 划分验证集（从训练集中切 10%）
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # 4. 训练模型
    print("\n[4/5] 训练模型 ...")
    model_names = args.models
    for name in model_names:
        train_model(name, X_tr, y_tr, X_val, y_val)

    # 5. 评估
    print("\n[5/5] 模型评估 ...")
    all_metrics = []
    roc_data = []

    for name in model_names:
        model = load(os.path.join(os.path.dirname(__file__), "models", f"{name}.joblib"))
        metrics = full_evaluation(
            model, X_test, y_test, y_test_orig, feature_cols, name,
            X_val=X_val, y_val=y_val,
        )
        all_metrics.append(metrics)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_data.append((name, y_test, y_prob))

    # ROC 对比
    if len(roc_data) > 0:
        plot_roc_curve(y_test, None, roc_data)

    # 指标对比
    if all_metrics:
        compare_models(all_metrics)

    print("\n" + "=" * 60)
    print("  训练完成！结果保存在 models/ 和 results/ 目录")
    print("=" * 60)


def cmd_eda(args):
    """探索性数据分析"""
    from data_loader import load_all_data
    from eda import (
        generate_eda_report, plot_label_distribution, plot_class_balance,
        plot_day_distribution, plot_top_features, plot_correlation_heatmap,
    )

    print("=" * 60)
    print("  探索性数据分析 (EDA)")
    print("=" * 60)

    print("\n加载数据 ...")
    df = load_all_data()

    generate_eda_report(df)

    print("\n生成图表 ...")
    plot_label_distribution(df)
    plot_class_balance(df)
    plot_day_distribution(df)
    plot_top_features(df, top_n=20)
    plot_correlation_heatmap(df)

    print("\nEDA 完成！所有图表已保存到 results/ 目录")


def cmd_predict(args):
    """对新数据进行预测"""
    from predict import predict_csv

    print("=" * 60)
    print("  恶意流量检测 - 推理预测")
    print("=" * 60)

    predict_csv(args.input, args.model, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="CIC-IDS2017 恶意流量检测系统"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eda", "predict"],
        help="运行模式: train (训练+评估), eda (数据分析), predict (推理预测)",
    )
    parser.add_argument(
        "--split", type=str, default="random",
        choices=["random", "time"],
        help="数据划分方式: random (随机分层采样, 默认), time (按时间划分)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3,
        help="测试集比例（仅 random 划分有效，默认 0.3）",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["lgb", "rf", "lr"],
        choices=["lgb", "rf", "lr"],
        help="要训练的模型，可指定多个，如 --models lgb rf",
    )
    parser.add_argument("--input", type=str, help="预测模式的输入 CSV 路径")
    parser.add_argument(
        "--model", type=str, default="lgb",
        choices=["lgb", "rf", "lr"],
        help="预测时使用的模型",
    )
    parser.add_argument("--output", type=str, help="预测结果的输出路径")

    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "eda":
        cmd_eda(args)
    elif args.mode == "predict":
        if not args.input:
            parser.error("predict 模式需要 --input 参数")
        cmd_predict(args)


if __name__ == "__main__":
    main()
