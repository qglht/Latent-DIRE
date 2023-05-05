import os
import sys
from argparse import ArgumentParser

import torch

from dire import LatentDIRE


def main(args, device: torch.device):
    '''Open images from read_dir, compute DIRE, and save to write_dir.
    The number of images loaded in at a time is determined by batch_size.
    '''
    model = LatentDIRE(device, args.model_id, use_fp16=False)
    for root, dirs, files in os.walk(args.read_dir):
        file_position = 0
        if not os.path.exists(args.write_dir):
            os.makedirs(args.write_dir)
        while True:
            # load batch of images
            batch = torch.cat([model.img_to_tensor(
                img) for img in [os.path.join(args.read_dir, file) for file in files[file_position:file_position+args.batch_size]]])
            # compute DIRE and latent DIRE
            with torch.no_grad():
                dire, latent_dire, _, _, _ = model(
                    batch.to(device), n_steps=50)

            # For testing locally
            # dire = torch.randn(args.batch_size, 3, 256, 256)
            # latent_dire = torch.randn(args.batch_size, 3, 64, 64)

            # save tensors
            for i in range(args.batch_size):
                torch.save(dire[i], os.path.join(
                    args.write_dir, f'{files[file_position+i]}_dire.pt'))
                torch.save(latent_dire[i], os.path.join(
                    args.write_dir, f'{files[file_position+i]}_latent_dire.pt'))
            # update file position
            file_position += args.batch_size
            if len(files) - file_position < args.batch_size:
                batch = torch.cat([model.img_to_tensor(
                    img) for img in [os.path.join(args.read_dir, file) for file in files[file_position:]]])
                with torch.no_grad():
                    dire, latent_dire, _, _, _ = model(
                        batch.to(device), n_steps=50)

                    # For testing locally
                    # dire = torch.randn(args.batch_size, 3, 256, 256)
                    # latent_dire = torch.randn(args.batch_size, 3, 64, 64)

                for i in range(len(files) - file_position):
                    torch.save(dire[i], os.path.join(
                        args.write_dir, f'{files[file_position+i]}_dire.pt'))
                    torch.save(latent_dire[i], os.path.join(
                        args.write_dir, f'{files[file_position+i]}_latent_dire.pt'))
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4",
                        help="model to use for computing DIRE")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="batch size for computing DIRE")
    parser.add_argument('--read_dir', type=str,
                        help="directory to read images from")
    parser.add_argument('--write_dir', type=str,
                        help="directory to write images to")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device)
