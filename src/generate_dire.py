from argparse import ArgumentParser
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
import wandb

# if this import doesn't work, you have not installed src, see https://www.notion.so/Docs-0dabc9ae19d54649b031e94e0cb0dff9
from src.dire import LatentDIRE


def main(args, device: torch.device):
    """Open images from read_dir, compute DIRE, and save to write_dir.
    The number of images loaded in at a time is determined by batch_size.
    """
    wandb.init(project="compute-dire", entity="latent-dire")
    logger = logging.getLogger(__name__)

    logger.info("Creating directories...")
    if not os.path.exists(args.write_dir_dire):
        os.makedirs(args.write_dir_dire)
    if not os.path.exists(args.write_dir_latent_dire):
        os.makedirs(args.write_dir_latent_dire)

    logger.info("Loading model...")
    model = LatentDIRE(device, pretrained_model_name=args.model_id, use_fp16=(True if device == "cuda" else False))

    dataset = ImageFolder("../data/data_dev_dire", transform=model.img_to_tensor)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    for idx, (batch, _) in tqdm(enumerate(dataloader)):
        batch = batch.squeeze(1)
        dire, latent_dire, *_ = model(batch.to(device))

        dire_path = os.path.join(args.write_dir_dire, f"{idx}_dire.pt")
        torch.save(dire, dire_path)
        latent_dire_path = os.path.join(args.write_dir_latent_dire, f"{idx}_latent_dire.pt")
        torch.save(dire, latent_dire_path)

        if idx % 100 == 0:
            logger.info(f"Processed {idx} batches ({idx * args.batch_size} images)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dev-run", action="store_true", help="Whether to run a test run.")
    parser.add_argument(
        "--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="model to use for computing DIRE"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for computing DIRE")
    parser.add_argument("--read_dir", type=str, help="directory to read images from")
    parser.add_argument("--write_dir_dire", type=str, help="directory to write dire to")
    parser.add_argument("--write_dir_latent_dire", type=str, help="directory to write latent dire to")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        filename="generate_dire.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args, device)
