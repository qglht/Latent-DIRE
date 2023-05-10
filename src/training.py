import argparse
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

import torch
import torch.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms.functional import hflip
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from data_loader import get_dataloaders
from nn.model_collection import MODEL_DICT


class Classifier(pl.LightningModule):
    def __init__(self, model: str, optimizer: str, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.classifier = MODEL_DICT[model]()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        dire, label = batch
        if np.random.rand() < 0.5:  # 50% chance for horizontal flip
            dire = hflip(dire)
        pred: torch.Tensor = self.model(dire)
        loss = F.binary_cross_entropy(pred, label)
        acc = accuracy_score(label, pred.argmax(axis=1))
        ap = average_precision_score(label, pred[:, 1])
        metrics = {"val_loss": loss, "val_acc": acc, "val_ap": ap}
        self.log(metrics)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        dire, label = batch
        pred: torch.Tensor = self.model(dire)
        loss = F.binary_cross_entropy(pred, label)
        acc = accuracy_score(label, pred.argmax(axis=1))
        ap = average_precision_score(label, pred[:, 1])
        metrics = {"val_loss": loss, "val_acc": acc, "val_ap": ap}
        self.log(metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        dire, label = batch
        pred: torch.Tensor = self.model(dire)
        loss = F.binary_cross_entropy(pred, label)
        acc = accuracy_score(label, pred.argmax(axis=1))
        ap = average_precision_score(label, pred[:, 1])
        metrics = {"val_loss": loss, "val_acc": acc, "val_ap": ap}
        self.log(metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam if self.hparams.optimizer == "Adam" else SGD
        if self.hparams.model == "resnet50_pixel":
            optimizer = optimizer(self.classifier.fc.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optimizer(self.classifier.parameters(), lr=self.hparams.learning_rate)

        lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_acc"}


def main(args: argparse.Namespace) -> None:
    seed_everything(33914, workers=True)

    # Setup Weights & Biases
    wandb_logger = WandbLogger(project="Training", entity="latent-dire", config=vars(args))

    # Load the data
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, shuffle=True)

    # Setup callbacks
    early_stop = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.0, patience=5, verbose=True)
    checkpoint = ModelCheckpoint(save_top_k=2, monitor="val_acc", mode="max", dirpath="models/")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    clf = Classifier(args.model, args.optimizer, args.learning_rate)
    trainer = Trainer(
        fast_dev_run=args.dev_run,  # uncomment to debug
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",  # use all available GPUs
        min_epochs=1,
        max_epochs=args.max_epochs,
        callbacks=[early_stop, checkpoint, lr_monitor],
        # deterministic=True,  # slower, but reproducable: https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
        precision="16-mixed",
        default_root_dir="models/",
        logger=wandb_logger,
    )
    trainer.fit(clf, train_loader, val_loader)
    trainer.test(clf, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_run", action="store_true", help="Whether to run a test run.")
    parser.add_argument("--model", type=str, default="resnet50_pixel")
    parser.add_argument("--latent", type=bool, default=False, help="Whether to use Latent DIRE")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--use_early_stopping", type=int, default=1, help="Whether to use early stopping.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_dir", type=str, default="/train")
    parser.add_argument("--val_dir", type=str, default="/val")
    parser.add_argument("--test_dir", type=str, default="/test")
    args = parser.parse_args()

    # Check arguments
    assert args.model in ["resnet50_latent", "resnet50_pixel", "mlp", "cnn"]
    assert args.batch_size > 0
    assert args.max_epochs > 0
    if args.latent:
        assert args.model in ["mlp", "cnn", "resnet50_latent"]
    else:
        assert args.model == "resnet50_pixel"

    main(args)
