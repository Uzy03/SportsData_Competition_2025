import os
import pandas as pd
from pathlib import Path


def print_feather(path_str: str, n_head: int = 5) -> None:
    path = Path(path_str)
    print(f"\n=== {path} ===")
    if not path.exists():
        print("[ERROR] ファイルが見つかりません")
        return
    try:
        df = pd.read_feather(path)
    except Exception as e:
        print(f"[ERROR] 読み込み失敗: {e}")
        return
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("dtypes:")
    print(df.dtypes)
    print("head:")
    print(df.head(n_head))


def main():
    base = "/Users/ujihara/m1_研究/SportsData_Competition_2025"
    meta_path = os.path.join(base, "Preprocessed_data/feather/meta/attack_2022091601_0001.feather")
    tensor_path = os.path.join(base, "Preprocessed_data/feather/tensor/attack_2022091601_0001.feather")

    print_feather(meta_path, n_head=10)
    print_feather(tensor_path, n_head=10)


if __name__ == "__main__":
    main()
