"""
模型训练模块
- LightGBM（主力模型）
- Random Forest（基线对照）
- Logistic Regression（快速基线）
"""
import os
import time
import json
import warnings
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """计算正负样本比例权重"""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **kwargs,
):
    """
    训练 LightGBM 二分类模型
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"  正负样本比例权重 (scale_pos_weight): {scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
    }
    # 允许覆盖参数
    params.update(kwargs)

    callbacks = []

    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        callbacks.append(lgb.early_stopping(20))
        callbacks.append(lgb.log_evaluation(0))  # 静默模式
    else:
        eval_set = [(X_train, y_train)]

    print(f"  训练 LightGBM ...")
    start = time.time()

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=callbacks if callbacks else None,
    )

    elapsed = time.time() - start
    print(f"  训练完成，耗时 {elapsed:.1f}s", flush=True)

    return model, elapsed


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **kwargs,
):
    """
    训练 Random Forest 模型
    """
    _ = X_val, y_val  # 统一接口，RF 不需要验证集
    scale_pos_weight = compute_scale_pos_weight(y_train)

    params = {
        "n_estimators": 50,
        "max_depth": 15,
        "min_samples_split": 100,
        "min_samples_leaf": 50,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
        "verbose": 0,
    }
    params.update(kwargs)

    print(f"  训练 Random Forest (n_estimators={params['n_estimators']}) ...", flush=True)
    start = time.time()

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"  训练完成，耗时 {elapsed:.1f}s", flush=True)

    return model, elapsed


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **kwargs,
):
    """
    训练 Logistic Regression 模型（快速基线）
    """
    _ = X_val, y_val  # 统一接口，LR 不需要验证集

    # 对大数据集采样以加速 LR 训练
    max_samples = 100000
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        print(f"  采样 {max_samples} 条用于训练 LR（共 {len(X_train)} 条）")
    else:
        X_sub, y_sub = X_train, y_train

    params = {
        "C": 1.0,
        "solver": "saga",
        "max_iter": 200,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
        "verbose": 0,
    }
    params.update(kwargs)

    print(f"  训练 Logistic Regression ...")
    start = time.time()

    model = LogisticRegression(**params)
    model.fit(X_sub, y_sub)

    elapsed = time.time() - start
    print(f"  训练完成，耗时 {elapsed:.1f}s", flush=True)

    return model, elapsed


def save_model(model, name: str, metadata: dict = None):
    """保存模型及元数据"""
    model_path = MODELS_DIR / f"{name}.joblib"
    dump(model, model_path)
    print(f"  模型已保存: {model_path}")

    if metadata:
        meta_path = MODELS_DIR / f"{name}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  元数据已保存: {meta_path}")


def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **kwargs,
):
    """
    统一训练入口
    支持: lgb, rf, lr
    """
    trainers = {
        "lgb": train_lightgbm,
        "rf": train_random_forest,
        "lr": train_logistic_regression,
    }

    if name not in trainers:
        raise ValueError(f"不支持的模型类型: {name}，可选: {list(trainers.keys())}")

    print(f"\n{'='*50}")
    print(f"训练模型: {name}")
    print(f"{'='*50}")

    model, elapsed = trainers[name](X_train, y_train, X_val, y_val, **kwargs)

    metadata = {
        "model": name,
        "train_samples": int(len(X_train)),
        "train_time_seconds": round(elapsed, 2),
        "features": int(X_train.shape[1]),
    }
    save_model(model, name, metadata)

    return model


if __name__ == "__main__":
    from data_loader import load_all_data, get_train_test_split
    from preprocessing import get_processed_train_test

    print("加载数据 ...")
    df = load_all_data()
    train_df, test_df = get_train_test_split(df)
    data = get_processed_train_test(train_df, test_df)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # LightGBM 用部分数据做验证集
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # 训练三个模型
    models = ["lgb", "rf", "lr"]
    for model_name in models:
        train_model(
            model_name,
            X_tr, y_tr,
            X_val, y_val,
        )
