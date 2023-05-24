import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch


async def pt_to_npz(pt_path: Path, dir: Path) -> None:
    arr = torch.load(pt_path, map_location="cpu").numpy()
    path = dir / pt_path.with_suffix(".npz").name
    np.savez(path, arr)


async def main(args):
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    file_list = Path(args.read_dir).iterdir()
    pt_list = [file for file in file_list if file.suffix == ".pt"]
    coros = [pt_to_npz(pt_path, write_dir) for pt_path in pt_list]

    semaphore = asyncio.Semaphore(1000)

    async def sem_task(task):
        async with semaphore:
            return await task

    await asyncio.gather(*(sem_task(coro) for coro in coros))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--read_dir", type=str, required=True)
    parser.add_argument("-w", "--write_dir", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
