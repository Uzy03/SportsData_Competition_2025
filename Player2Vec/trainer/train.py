import argparse
import pytorch_lightning as pl
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
    trainer = pl.Trainer(max_epochs=args.epochs, precision=precision, accelerator="auto", log_every_n_steps=1)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()