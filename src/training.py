import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from classifier import build_classifier, compute_dire
from data import load_data


def train_loop(
        train_dataloader: DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> None:
    for batch, (image, label) in tqdm(enumerate(train_dataloader)):
        # Compute prediction error
        dire = compute_dire(image)
        pred = model(dire)
        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()


def train_model(
        train_dataloader: DataLoader, 
        model: torch.nn.Module, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epochs: int = 5) -> None:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the data
    train_loader, test_loader = load_data()

    # Build the model
    sample_layers = [3, 64, 64, 2]
    model = build_classifier(device, model_name='mlp', layers=sample_layers)

    # Train the model
    loss_fn = nn.BCELoss()
    # Or Adam, to be discussed
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_model(train_loader, model, loss_fn, optimizer, epochs=10)

    # Save the model
    # save_model(model)
