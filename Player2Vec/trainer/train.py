import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datamodules.datamodule import SoccerDataModule
from models.model import SoccerLightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", nargs="?", help="compatibility placeholder")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    dm = SoccerDataModule(args.data, batch_size=args.batch)
    model = SoccerLightningModule()
    precision = "16-mixed" if args.fp16 else 32
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="soccer_model_{epoch:02d}_{val_loss:.4f}",
        save_last=True,
        monitor=None,
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=precision,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[checkpoint_cb],
        default_root_dir=ckpt_dir,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()