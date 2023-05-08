from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from data import load_data
from models.cnn import build_cnn
from models.mlp import build_mlp
from models.resnet50 import (build_resnet50_latent, build_resnet50_pixel,
                             preprocess_resnet50_latent,
                             preprocess_resnet50_pixel)


def train_model(
    train_dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
) -> None:
    for epoch in range(epochs):
        pred_hist, label_hist = [], []
        for batch, (dire, label) in tqdm(enumerate(train_dataloader)):
            dire = dire.to(device)
            label: torch.Tensor = label.to(device)

            # Compute prediction error
            pred: torch.Tensor = model(dire)
            loss = loss_fn(pred, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_hist.append(pred.detach().cpu().numpy())
            label_hist.append(label.detach().cpu().numpy())


        # Compute metrics
        pred_hist = np.concatenate(pred_hist, axis=0)
        label_hist = np.concatenate(label_hist, axis=0)
        acc = accuracy_score(label_hist, pred_hist.argmax(axis=1))
        ap = average_precision_score(label_hist, pred_hist[:, 1])
        wandb.log({"accuracy": acc, "average_precision": ap})
        wandb.log({"loss": loss.item()})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent", type=bool, default=False,
                        help="Whether to use Latent DIRE")
    parser.add_argument("--optimizer", type=str,
                        default="sgd", help="Optimizer to use")
    args = parser.parse_args()

    # Check arguments
    assert args.model in ["resnet50_latent", "resnet50_pixel", "mlp", "cnn"]
    assert args.batch_size > 0
    assert args.epochs > 0
    if args.latent:
        assert args.model in ["mlp", "cnn", "resnet50_latent"]
    else:
        assert args.model == "resnet50_pixel"

    # Setup Weights & Biases
    wandb.login()
    wandb.init(project="dire", entity="dire")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(0)

    # Load the data
    train_loader, test_loader = load_data()

    # Build the model
    model_dict = {
        "resnet50_latent": build_resnet50_latent,
        "resnet50_pixel": build_resnet50_pixel,
        "mlp": build_mlp,
        "cnn": build_cnn,
    }
    model = model_dict[args.model]().to(device)

    # Train the model
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    if args.model == "resnet50_pixel":
        optimizer = torch.optim.Adam(model.fc.parameters())
    train_model(train_loader, model,
                loss_fn, optimizer, epochs=10)

    # Close Weights & Biases
    wandb.finish()

    # Save the model weights
    torch.save(model.state_dict(), f"../model_weights/{args.model}.pt")
