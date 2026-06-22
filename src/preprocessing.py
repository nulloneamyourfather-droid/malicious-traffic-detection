"""
特征工程模块：
- 特征选择
- 标准化
- 训练/测试集划分
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 需要丢弃的非特征列
NON_FEATURE_COLUMNS = {"Label", "Label_Binary", "Day"}

# 全是零的列（Bulk 相关，占位符）
ZERO_VARIANCE_COLUMNS = [
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# 重复列
DUPLICATE_COLUMNS = ["Fwd Header Length.1"]


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列列表（排除标签等非特征列）"""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def remove_low_variance_features(
    df: pd.DataFrame, threshold: float = 0.0
) -> tuple:
    """
    去除低方差的特征（方差为 0 的特征）
    返回 (清洗后的 df, 保留的特征列表)
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    # 计算方差
    variances = X.var()

    # 找出方差 > threshold 的特征
    keep_cols = variances[variances > threshold].index.tolist()
    drop_cols = variances[variances <= threshold].index.tolist()

    if drop_cols:
        print(f"  已去除 {len(drop_cols)} 个零/低方差特征: {drop_cols}")

    return keep_cols, drop_cols


def preprocess_data(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: StandardScaler = None,
) -> tuple:
    """
    特征工程主流程

    Parameters:
        df: 输入 DataFrame（包含特征和标签）
        fit_scaler: 是否拟合新的 Scaler（训练集=True，测试集=False）
        scaler: 已有的 Scaler（测试集时传入训练集的 scaler）

    Returns:
        (X_scaled, y_binary, y_multiclass, scaler, feature_cols)
    """
    # 排除非特征列
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y_binary = (df["Label_Binary"] == "Malicious").astype(int).values
    y_multiclass = df["Label"].values

    # 处理无穷值和 NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    print(f"\n特征矩阵: {X.shape[0]} 行 x {X.shape[1]} 列")

    # 标准化
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, MODELS_DIR / "scaler.joblib")
        print(f"  Scaler 已保存到 {MODELS_DIR / 'scaler.joblib'}")
    else:
        if scaler is None:
            raise ValueError("fit_scaler=False 时必须提供 scaler 参数")
        X_scaled = scaler.transform(X)

    return X_scaled, y_binary, y_multiclass, scaler, feature_cols


def get_processed_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    一站式处理训练集和测试集：
    - 特征筛选
    - 标准化（训练集 fit, 测试集 transform）
    - 返回处理后的数据
    """
    # 训练集特征预处理 + 拟合 scaler
    X_train, y_train, y_train_orig, scaler, feature_cols = preprocess_data(
        train_df, fit_scaler=True
    )

    # 测试集使用相同的 scaler
    X_test, y_test, y_test_orig, _, _ = preprocess_data(
        test_df, fit_scaler=False, scaler=scaler
    )

    # 保存特征列名
    feature_list_path = MODELS_DIR / "feature_columns.txt"
    with open(feature_list_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_orig": y_train_orig,
        "y_test_orig": y_test_orig,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from data_loader import load_all_data, get_train_test_split

    df = load_all_data()
    train_df, test_df = get_train_test_split(df)
    data = get_processed_train_test(train_df, test_df)

    print(f"\n处理完成:")
    print(f"  训练集: {data['X_train'].shape}")
    print(f"  测试集: {data['X_test'].shape}")
    print(f"  特征数: {len(data['feature_cols'])}")
