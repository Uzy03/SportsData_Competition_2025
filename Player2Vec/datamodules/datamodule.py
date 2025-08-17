import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
import ast

class SoccerDataset(Dataset):
    """Return (patch_tensor, player_ids, team_id).
    If csv_path does not exist, random synthetic data are generated.
    patch_tensor shape: (23, 150, 2)
    """
    def __init__(self, csv_path: str | None = None, window: int = 150, length: int = 1000):
        self.window = window
        self.csv_path = csv_path
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.length = len(self.df)
        else:
            self.df = None
            self.length = length

    def __len__(self):
        return self.length

    def _load_patch(self, path: str):
        try:
            arr = np.load(path)
            return arr.astype(np.float32)
        except Exception:
            return np.zeros((23, self.window, 2), dtype=np.float32)

    def __getitem__(self, idx: int):
        if self.df is not None:
            row = self.df.iloc[idx]
            patch = self._load_patch(row.get("path", ""))
            player_ids = row.get("player_ids", "")
            if isinstance(player_ids, str):
                player_ids = [int(x) for x in player_ids.split(',') if x]
            player_ids = player_ids[:23]
            while len(player_ids) < 23:
                player_ids.append(0)
            team_id = int(row.get("team_id", 0))
        else:
            patch = np.zeros((23, self.window, 2), dtype=np.float32)
            team_id = idx % 2
            patch[0] += team_id  # embed team info for training demo
            player_ids = list(range(23))
            player_ids[0] = team_id
        # zero pad / crop
        T = min(self.window, patch.shape[1])
        if patch.shape[1] < self.window:
            pad = np.zeros((23, self.window, 2), dtype=np.float32)
            pad[:, :patch.shape[1]] = patch[:, :T]
            patch = pad
        else:
            patch = patch[:, :T]
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        player_ids = torch.tensor(player_ids, dtype=torch.long)
        team_id = torch.tensor(team_id, dtype=torch.long)
        return patch_tensor, player_ids, team_id

class PreprocessedSoccerDataset(Dataset):
    """Preprocessed_dataのparquetファイルからデータを読み込むデータセット"""
    
    def __init__(self, data_dir="Preprocessed_data/parquet", window: int = 150, max_files=None, fps: int = 5):
        self.data_dir = Path(data_dir)
        self.tensor_dir = self.data_dir / "tensor"
        self.meta_dir = self.data_dir / "meta"
        self.window = window
        self.fps = fps  # 目標fps
        self.fps_ratio = 25 // fps  # 25fpsから目標fpsへの間隔
        
        # 利用可能なファイル一覧を取得
        self.available_files = []
        for meta_file in self.meta_dir.glob("*.parquet"):
            filename = meta_file.stem
            tensor_file = self.tensor_dir / f"{filename}.parquet"
            if tensor_file.exists():
                self.available_files.append(filename)
        
        if max_files:
            self.available_files = self.available_files[:max_files]
        
        print(f"利用可能なデータファイル数: {len(self.available_files)}")
        print(f"目標fps: {fps} (25fps → {fps}fps, 間隔: {self.fps_ratio})")
        
        # メタデータを事前に読み込み
        self.meta_data = {}
        for filename in self.available_files:
            meta_path = self.meta_dir / f"{filename}.parquet"
            meta_df = pd.read_parquet(meta_path)
            self.meta_data[filename] = meta_df.iloc[0].to_dict()
    
    def __len__(self):
        return len(self.available_files)
    
    def __getitem__(self, idx):
        filename = self.available_files[idx]
        meta = self.meta_data[filename]
        
        # テンソルデータの読み込み
        tensor_path = self.tensor_dir / f"{filename}.parquet"
        tensor_df = pd.read_parquet(tensor_path)
        
        # テンソルの再構築
        shape = (meta['tensor_shape_0'], meta['tensor_shape_1'], meta['tensor_shape_2'])
        tensor = np.full(shape, np.nan, dtype=np.float32)
        
        for _, row in tensor_df.iterrows():
            entity_idx = int(row['entity_idx'])
            frame_idx = int(row['frame_idx'])
            tensor[entity_idx, frame_idx, 0] = row['x']
            tensor[entity_idx, frame_idx, 1] = row['y']
        
        # NaNを0で埋める
        tensor = np.nan_to_num(tensor, nan=0.0)
        
        # フレーム間隔調整: 5fps相当にする
        original_frames = tensor.shape[1]
        
        # 常に(23, 150, 2)の形状を保証
        sampled_tensor = np.zeros((23, self.window, 2), dtype=np.float32)
        
        if original_frames >= self.fps_ratio:
            # 5フレームごとにサンプリング
            sampled_indices = np.arange(0, original_frames, self.fps_ratio)
            if len(sampled_indices) > self.window:
                # 150フレームを超える場合は、最後の150フレームを取得
                sampled_indices = sampled_indices[-self.window:]
            
            # サンプリングされたフレームでテンソルを再構築
            for i, frame_idx in enumerate(sampled_indices):
                if i < self.window and frame_idx < original_frames:
                    sampled_tensor[:, i, :] = tensor[:, frame_idx, :]
        else:
            # 元のフレーム数が少ない場合は、利用可能な分だけコピー
            copy_frames = min(original_frames, self.window)
            sampled_tensor[:, :copy_frames, :] = tensor[:, :copy_frames, :]
        
        # 座標データの正規化: [-1, 1]の範囲に
        # サッカーコートの座標範囲を想定（X: ±52.5, Y: ±34）
        # より安全な範囲で正規化
        x_coords = sampled_tensor[:, :, 0]
        y_coords = sampled_tensor[:, :, 1]
        
        # 座標範囲の計算（0でない値のみ）
        x_nonzero = x_coords[x_coords != 0]
        y_nonzero = y_coords[y_coords != 0]
        
        if len(x_nonzero) > 0 and len(y_nonzero) > 0:
            x_min, x_max = x_nonzero.min(), x_nonzero.max()
            y_min, y_max = y_nonzero.min(), y_nonzero.max()
            
            # 正規化範囲の決定（データの範囲 + マージン）
            x_range = max(abs(x_min), abs(x_max)) * 1.1  # 10%マージン
            y_range = max(abs(y_min), abs(y_max)) * 1.1
            
            # ゼロ除算を避ける
            if x_range > 0:
                x_mask = x_coords != 0
                sampled_tensor[:, :, 0] = np.where(x_mask, x_coords / x_range, 0)
            
            if y_range > 0:
                y_mask = y_coords != 0
                sampled_tensor[:, :, 1] = np.where(y_mask, y_coords / y_range, 0)
        
        # 形状を最終確認
        assert sampled_tensor.shape == (23, self.window, 2), f"Unexpected tensor shape: {sampled_tensor.shape}"
        
        # 実ラベル: meta['entities'] から選手ID配列を生成
        entities_raw = meta.get('entities', '[]')
        try:
            entities_list = ast.literal_eval(entities_raw)
        except Exception:
            entities_list = []
        
        # 23長に合わせ、非整数や範囲外は0に置換
        player_ids_list = []
        for i in range(23):
            if i < len(entities_list):
                v = entities_list[i]
                if isinstance(v, int) and 0 <= v <= 999999:  # より広い範囲で許可
                    player_ids_list.append(v)
                elif v == 'ball':
                    player_ids_list.append(-1)  # ballは-1として扱う
                else:
                    player_ids_list.append(0)
            else:
                player_ids_list.append(0)
        
        # チームIDはメタに無いため暫定0
        team_id_val = 0
        
        # PyTorchテンソルに変換 - 形状を確認
        patch_tensor = torch.tensor(sampled_tensor, dtype=torch.float32)
        player_ids = torch.tensor(player_ids_list, dtype=torch.long)
        team_id = torch.tensor(team_id_val, dtype=torch.long)
        
        # 最終的な形状チェック
        assert patch_tensor.shape == (23, self.window, 2), f"patch_tensor shape mismatch: {patch_tensor.shape}"
        assert player_ids.shape == (23,), f"player_ids shape mismatch: {player_ids.shape}"
        assert team_id.shape == (), f"team_id shape mismatch: {team_id.shape}"
        
        return patch_tensor, player_ids, team_id

class SoccerDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str | None = None, batch_size: int = 64, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                 random_seed: int = 42, window: int = 150):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.window = window
        
        # 比率の合計が1.0になることを確認
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比率の合計が1.0になる必要があります。現在: {total_ratio}")

    def setup(self, stage: str | None = None):
        # 全データセットの作成
        full_dataset = SoccerDataset(self.data_path, window=self.window)
        
        # データセットの長さを取得
        total_length = len(full_dataset)
        
        # 各分割のサイズを計算
        train_size = int(total_length * self.train_ratio)
        val_size = int(total_length * self.train_ratio)
        test_size = total_length - train_size - val_size  # 残りをテスト用に
        
        # データセットを分割
        torch.manual_seed(self.random_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size]
        )
        
        print(f"データセット分割完了:")
        print(f"  トレーニング: {len(self.train_dataset)} ({self.train_ratio*100:.1f}%)")
        print(f"  検証: {len(self.val_dataset)} ({self.val_ratio*100:.1f}%)")
        print(f"  テスト: {len(self.test_dataset)} ({self.test_ratio*100:.1f}%)")

    def train_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
    
    def val_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
    
    def test_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)

    def _loader_kwargs(self):
        """Return DataLoader kwargs depending on GPU availability."""
        if torch.cuda.is_available():
            return {"num_workers": 12, "pin_memory": True}
        else:
            return {"num_workers": 0, "pin_memory": False}

class PreprocessedSoccerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="Preprocessed_data/parquet", batch_size=32, 
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                 random_seed=42, window=150, max_files=None, fps: int = 5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.window = window
        self.max_files = max_files
        self.fps = fps  # 目標fps
        self.fps_ratio = 25 // fps  # 25fpsから目標fpsへの間隔
        
        # 比率の合計が1.0になることを確認
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比率の合計が1.0になる必要があります。現在: {total_ratio}")
    
    def setup(self, stage=None):
        # 全データセットの作成
        full_dataset = PreprocessedSoccerDataset(self.data_dir, self.window, self.max_files, self.fps)
        
        # データセットの長さを取得
        total_length = len(full_dataset)
        
        # 各分割のサイズを計算
        train_size = int(total_length * self.train_ratio)
        val_size = int(total_length * self.val_ratio)
        test_size = total_length - train_size - val_size
        
        # データセットを分割
        torch.manual_seed(self.random_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        print(f"データセット分割完了:")
        print(f"  トレーニング: {len(self.train_dataset)} ({self.train_ratio*100:.1f}%)")
        print(f"  検証: {len(self.val_dataset)} ({self.val_ratio*100:.1f}%)")
        print(f"  テスト: {len(self.test_dataset)} ({self.test_ratio*100:.1f}%)")
    
    def train_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            **kwargs
        )
    
    def val_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            **kwargs
        )
    
    def test_dataloader(self):
        kwargs = self._loader_kwargs()
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            **kwargs
        )

    def _loader_kwargs(self):
        """Return DataLoader kwargs depending on GPU availability."""
        if torch.cuda.is_available():
            return {"num_workers": 12, "pin_memory": True}
        else:
            return {"num_workers": 0, "pin_memory": False}