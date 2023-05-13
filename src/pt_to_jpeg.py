import argparse
import os
from pathlib import Path
from PIL import Image

import torch
from tqdm import tqdm


def tensor_to_pil(image: torch.Tensor):
    if image.dim == 3:
        image = image.unsqueeze(0)

    image = ((image + 1) * 127.5).clamp(0, 255).to(dtype=torch.uint8)  # [-1, 1] to [0, 255]
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    if image.shape[-1] == 1:
        # special case for grayscale (single channel) image
        pil_image = [Image.fromarray(image.squeeze(), mode="L") for image in image]
    else:
        pil_image = [Image.fromarray(image) for image in image]

    return pil_image


def convert_batch(batch: Path, dir: Path) -> None:
    t = torch.load(batch, map_location="cpu")
    pil_list = tensor_to_pil(t)
    batch_idx = int(batch.stem.split("_")[0])
    batch_size = t.shape[0]
    for i, pil in enumerate(pil_list):
        pil.convert("RGB")
        path = Path(dir) / f"{batch_size * batch_idx + i}.jpg"
        pil.save(path)


def main(args):
    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)
    batches: Path = Path(args.read_dir).iterdir()
    write_dir: Path = Path(args.write_dir)
    for batch in tqdm(batches):
        convert_batch(batch, write_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_dir", type=str, required=True)
    parser.add_argument("--write_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
