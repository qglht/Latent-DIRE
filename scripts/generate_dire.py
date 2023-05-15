"""
This script computes both the DIRE and latent DIRE representations for a folder of images.

It assumes you have a folder of images you want to compute the representations for. The folder should be 
structured as follows:

    read_dir
    ├──img1.JPEG
    ├──img2.JPEG
    ├──img3.JPG
    :

Before launching the script, compress your folder using 

    tar cf compressed_name.tar read_dir

and put it at /cluster/scratch/user/ where user is your ETH Kürzel. Finally, make sure you adapt the dire_generation.sh slurm script. 
You need to provide the path of your compressed folder in the variable COMPRESSED_FOLDER_PATH.
"""


from argparse import ArgumentParser
from functools import partial
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import wandb

# if this import doesn't work, you have not installed src, see https://www.notion.so/Docs-0dabc9ae19d54649b031e94e0cb0dff9
from src.dire import LatentDIRE, ADMDIRE


def main(args, device: torch.device):
    """Open images from read_dir, compute DIRE, and save to write_dir.
    The number of images loaded in at a time is determined by batch_size.
    """
    wandb.init(project="compute-dire", entity="latent-dire")
    logger = logging.getLogger(__name__)

    logger.info("Loading model...")
    latent = args.model_id in ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]
    if latent:
        model = LatentDIRE(
            device,
            pretrained_model_name=args.model_id,
            use_fp16=(True if device == "cuda" else False),
        )
        transform = partial(model.img_to_tensor, size=512)
    else:
        model = ADMDIRE(
            device,
            model_path=args.model_id,
            use_fp16=(True if device == "cuda" else False),
        )
        transform = partial(model.img_to_tensor, size=256)

    scratch_dir = os.environ["TMPDIR"]
    img_dir = f"{scratch_dir}/images/"
    dataset = ImageFolder(img_dir, transform=transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    # create directories if they don't exist
    write_dir_dire = Path(args.write_dir_dire)
    write_dir_ldire = Path(args.write_dir_ldire)
    write_dir_decoded_ldire = Path(args.write_dir_decoded_ldire)
    write_dir_dire.mkdir(parents=True, exist_ok=True)
    write_dir_ldire.mkdir(parents=True, exist_ok=True)
    write_dir_decoded_ldire.mkdir(parents=True, exist_ok=True)

    logger.info("Computing DIRE...")
    for batch_idx, (batch, _) in tqdm(enumerate(dataloader)):
        batch = batch.squeeze(1).to(device)
        if latent:
            dire, ldire, *_ = model(batch, n_steps=args.ddim_steps)
            decoded_ldire = model.decode(ldire)
        else:
            dire, _ = model(batch, n_steps=args.ddim_steps)

        dire = model.tensor_to_pil(dire)
        for i in range(args.batch_size):
            dire_path = write_dir_dire / f"{batch_idx*args.batch_size + i}_dire.jpeg"
            dire[i].save(dire_path)
            if latent:
                ldire_path = write_dir_ldire / f"{batch_idx*args.batch_size + i}_ldire.pt"
                torch.save(ldire[i], ldire_path)
                decoded_ldire_path = write_dir_decoded_ldire / f"{batch_idx*args.batch_size + i}_decoded_ldire.jpeg"
                decoded_ldire[i].save(decoded_ldire_path)

        if args.dev_run:
            break

        if batch_idx % 10 == 0:
            logger.info(f"Processed {batch_idx} batches ({batch_idx * args.batch_size} images)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dev-run", action="store_true", help="Whether to run a test run.")
    parser.add_argument(
        "--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="model to use for computing DIRE"
    )
    parser.add_argument("--ddim_steps", type=int, required=True, help="How many DDIM steps to take.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for computing DIRE")
    parser.add_argument("--write_dir_dire", type=str, help="directory to write dire to")
    parser.add_argument("--write_dir_ldire", type=str, help="directory to write latent dire to")
    parser.add_argument("--write_dire_decoded_ldire", type=str, help="directory to write decoded latent dire to")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.model_id in [
        "CompVis/stable-diffusion-v1-4",
        "runwayml/stable-diffusion-v1-5",
        "models/lsun_bedroom.pt",
    ]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    main(args, device)
