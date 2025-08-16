import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import argparse

def main():
    parser = argparse.ArgumentParser(description="Soccer Player2Vec モデルの学習")
    parser.add_argument("--data_dir", type=str, default="Preprocessed_data/parquet", 
                       help="データディレクトリのパス")
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大エポック数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--max_files", type=int, default=None, help="使用する最大ファイル数（Noneで全ファイル）")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="トレーニングデータの比率")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="検証データの比率")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="テストデータの比率")
    parser.add_argument("--devices", type=int, default=1, help="使用するデバイス数")
    
    args = parser.parse_args()
    
    # データモジュールの初期化
    print("データモジュールを初期化中...")
    from datamodules.datamodule import PreprocessedSoccerDataModule
    data_module = PreprocessedSoccerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_files=args.max_files
    )
    
    # モデルの初期化
    print("モデルを初期化中...")
    from models.model import SoccerLightningModule
    model = SoccerLightningModule(
        num_players=100,  # 選手数
        num_teams=2,      # チーム数
        lr=args.lr
    )
    
    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="soccer_model_{epoch:02d}_{val_loss:.4f}",
            save_top_k=3,
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ]
    
    # ロガーの設定（CSV形式）
    logger = CSVLogger("logs", name="soccer_player2vec")
    
    # トレーナーの設定
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.devices
        precision = "16-mixed"
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        precision=precision,
    )
    
    # 学習の実行
    print("学習を開始します...")
    trainer.fit(model, datamodule=data_module)
    
    # テストの実行
    print("テストを実行します...")
    trainer.test(model, datamodule=data_module)
    
    print("学習完了！")

if __name__ == "__main__":
    main()
