"""
CIC-IDS2017 数据加载与预处理模块
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 数据路径常量
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "MachineLearningCSV" / "MachineLearningCVE"
PROCESSED_DIR = DATA_DIR / "processed"

# 8 个 CSV 文件按时间顺序排列（用于时间划分）
FILE_CONFIG = [
    # (文件名, 日期标签)
    ("Monday-WorkingHours.pcap_ISCX.csv", "monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "wednesday"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "thursday"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "thursday"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "friday"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "friday"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "friday"),
]

# 标签映射（原始标签 → 二分类标签）
ATTACK_LABELS = {
    "FTP-Patator", "SSH-Patator",
    "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed",
    "Web Attack � Brute Force", "Web Attack � XSS", "Web Attack � Sql Injection",
    "Infiltration", "Bot", "PortScan", "DDoS",
}

# 需要丢弃的列（重复列或无意义的列）
DROP_COLUMNS = [" Fwd Header Length.1", "Fwd Header Length.1"]

# 全为 Bulk 相关列（原始论文中为占位符，全为 0）
BULK_COLUMNS = [
    " Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", " Bwd Avg Bulk Rate",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名：去除前后空格"""
    df.columns = [col.strip() for col in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    - 去除全 NaN 的行
    - 填充 NaN 为 0
    - 填充 Inf 为 0
    - 去除重复 header 行（内容恰好是列名的行）
    """
    # 去除第一行如果是 header 重复的情况（某些文件有）
    label_col = [c for c in df.columns if c.lower() == "label"][0]
    df = df[df[label_col] != label_col].copy()

    # 去除全 NaN 行
    df = df.dropna(how="all")

    # 只处理数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

    return df


def make_binary_label(label_series: pd.Series) -> pd.Series:
    """将原始标签转为二分类标签：BENIGN / Malicious"""
    return label_series.str.strip().apply(
        lambda x: "BENIGN" if x.strip() == "BENIGN" else "Malicious"
    )


def load_single_file(filepath: str, day_label: str) -> pd.DataFrame:
    """加载单个 CSV 文件，添加日期列"""
    print(f"  加载 {Path(filepath).name} ...")
    df = pd.read_csv(filepath)
    df = clean_column_names(df)
    df = clean_data(df)

    # 添加原始标签（去除空格）
    label_col = "Label"
    df["Label"] = df[label_col].str.strip()

    # 添加二分类标签
    df["Label_Binary"] = make_binary_label(df["Label"])

    # 添加日期标记
    df["Day"] = day_label

    return df


def load_all_data(force_reload: bool = False) -> pd.DataFrame:
    """
    加载全部 8 个 CSV 文件并合并。
    如果已存在缓存的 parquet 文件，直接读取加速。
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cache_path = PROCESSED_DIR / "cic_ids2017_full.parquet"

    if cache_path.exists() and not force_reload:
        print(f"发现缓存文件，直接加载: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"已加载，共 {len(df)} 条记录，{len(df.columns)} 列")
        return df

    all_dfs = []
    for filename, day_label in FILE_CONFIG:
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f"  警告: 文件不存在，跳过 {filepath}")
            continue
        df = load_single_file(str(filepath), day_label)
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"未找到任何 CSV 文件，请检查路径: {RAW_DIR}")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并完成: 共 {len(df)} 条记录，{len(df.columns)} 列")

    # 删除重复列
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  已删除重复列: {cols_to_drop}")

    # 缓存到 parquet
    print(f"  缓存数据到 {cache_path} ...")
    df.to_parquet(cache_path, index=False)
    print("  缓存完成")

    return df


def get_train_test_split(
    df: pd.DataFrame,
    train_days: tuple = ("monday", "tuesday", "wednesday", "thursday"),
    test_days: tuple = ("friday",),
) -> tuple:
    """
    按时间划分训练集和测试集。
    默认：Mon-Thu → 训练集，Fri → 测试集
    """
    train = df[df["Day"].isin(train_days)].copy()
    test = df[df["Day"].isin(test_days)].copy()

    print(f"\n时间划分结果:")
    print(f"  训练集: {', '.join(train_days)} → {len(train)} 条")
    print(f"  测试集: {', '.join(test_days)} → {len(test)} 条")

    attack_ratio_train = (train["Label_Binary"] == "Malicious").mean() * 100
    attack_ratio_test = (test["Label_Binary"] == "Malicious").mean() * 100
    print(f"  训练集攻击占比: {attack_ratio_train:.2f}%")
    print(f"  测试集攻击占比: {attack_ratio_test:.2f}%")

    return train, test


def get_random_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """
    随机分层采样划分训练集和测试集。
    按原始标签分层，确保所有攻击类型在训练和测试中都出现。
    """
    from sklearn.model_selection import train_test_split as tts

    train, test = tts(
        df, test_size=test_size, random_state=random_state,
        stratify=df["Label"],  # 按原始标签分层
    )

    print(f"\n随机分层采样划分结果 (test_size={test_size}):")
    print(f"  训练集: {len(train)} 条")
    print(f"  测试集: {len(test)} 条")

    attack_ratio_train = (train["Label_Binary"] == "Malicious").mean() * 100
    attack_ratio_test = (test["Label_Binary"] == "Malicious").mean() * 100
    print(f"  训练集攻击占比: {attack_ratio_train:.2f}%")
    print(f"  测试集攻击占比: {attack_ratio_test:.2f}%")

    return train, test


if __name__ == "__main__":
    # 测试加载
    df = load_all_data()
    print(f"\n标签分布 (二分类):")
    print(df["Label_Binary"].value_counts())
    print(f"\n标签分布 (原始):")
    print(df["Label"].value_counts())
    print(f"\n日期分布:")
    print(df["Day"].value_counts())

    train, test = get_train_test_split(df)
