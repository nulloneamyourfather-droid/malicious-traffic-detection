"""
推理预测模块：
- 加载训练好的模型和 Scaler
- 对新的 CSV 流量数据进行预测
"""
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(model_name: str = "lgb"):
    """
    加载训练好的模型和相关文件

    Parameters:
        model_name: lgb / rf / lr

    Returns:
        (model, scaler, feature_cols)
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    features_path = MODELS_DIR / "feature_columns.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}，请先运行训练: python main.py --mode train")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler 不存在: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"特征列表不存在: {features_path}")

    model = load(model_path)
    scaler = load(scaler_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"模型: {model_name}")
    print(f"特征数: {len(feature_cols)}")

    return model, scaler, feature_cols


def predict_csv(
    input_path: str,
    model_name: str = "lgb",
    output_path: str = None,
):
    """
    对 CSV 文件中的流量数据进行预测

    Parameters:
        input_path: 输入 CSV 文件路径（必须包含模型所需的特征列）
        model_name: 使用的模型
        output_path: 输出 CSV 路径（不指定则自动生成）
    """
    # 加载模型
    model, scaler, feature_cols = load_model(model_name)

    # 读取数据
    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"共 {len(df)} 条记录，{len(df.columns)} 列")

    # 列名清理
    df.columns = [c.strip() for c in df.columns]

    # 检查特征完整性
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"警告: 缺少 {len(missing_features)} 个特征列:")
        for f in missing_features[:10]:
            print(f"  - {f}")
        print(f"缺失特征将被填充为 0")

    # 准备特征矩阵
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col].values
        else:
            X[col] = 0.0

    # 处理 NaN 和 Inf
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # 结果
    df["Prediction"] = ["Malicious" if p == 1 else "BENIGN" for p in y_pred]
    df["Malicious_Probability"] = np.round(y_prob, 4)

    # 统计
    n_malicious = (y_pred == 1).sum()
    n_benign = (y_pred == 0).sum()
    print(f"\n预测结果:")
    print(f"  BENIGN:   {n_benign} ({n_benign / len(df) * 100:.1f}%)")
    print(f"  Malicious: {n_malicious} ({n_malicious / len(df) * 100:.1f}%)")

    # 输出
    if output_path is None:
        input_name = Path(input_path).stem
        output_path = Path(input_path).parent / f"{input_name}_predicted.csv"

    df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")

    return df


def predict_single(features: dict, model_name: str = "lgb") -> dict:
    """
    对单条流量特征进行预测

    Parameters:
        features: 特征字典 {feature_name: value, ...}
        model_name: 模型名

    Returns:
        {"prediction": "BENIGN"/"Malicious", "probability": 0.xx}
    """
    model, scaler, feature_cols = load_model(model_name)

    # 构建特征向量
    X = pd.DataFrame([features])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]
    pred = "Malicious" if prob >= 0.5 else "BENIGN"

    return {"prediction": pred, "probability": round(float(prob), 4)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="恶意流量检测 - 推理")
    parser.add_argument("--input", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "rf", "lr"])
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径")

    args = parser.parse_args()

    predict_csv(args.input, args.model, args.output)
