import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from datasets.nuplan_dataset import NuplanDataset


def _load_sample(dataset: NuplanDataset, idx: int) -> Optional[Tuple[str, str, str]]:
    """Attempt to load a single sample; return error info on failure."""
    path = dataset.processed_paths[idx]
    name = (
        dataset.processed_file_names[idx]
        if idx < len(dataset.processed_file_names)
        else Path(path).name
    )
    try:
        HeteroData(torch.load(path))
    except Exception as exc:  # pylint: disable=broad-except
        return name, path, repr(exc)
    return None


def check_dataset(split_name: str, dataset: NuplanDataset, num_workers: int) -> List[Tuple[str, str, str]]:
    """Load every entry in a dataset; return a list of corrupted files."""
    total = len(dataset)
    print(f"[{split_name}] Checking {total} files with {num_workers} workers...")

    corrupted: List[Tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load_sample, dataset, idx) for idx in range(total)]
        with tqdm(total=total, desc=f"[{split_name}] Checking files", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    corrupted.append(result)
                pbar.update()

    if corrupted:
        print(f"[{split_name}] Found {len(corrupted)} corrupted files.")
    else:
        print(f"[{split_name}] All files loaded successfully.")
    return corrupted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify corrupted processed NuPlan files.")
    parser.add_argument("--root", required=True, help="Path to the NuPlan dataset root directory.")
    parser.add_argument("--train-metadata", dest="train_metadata", required=True, help="Path to training metadata JSON.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of threads to use when loading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_workers = max(1, args.num_workers)

    datasets_to_check = [
        ("train", NuplanDataset(args.root, args.train_metadata, transform=None)),
    ]

    all_corrupted: List[Tuple[str, str, str]] = []
    for split_name, dataset in datasets_to_check:
        corrupted = check_dataset(split_name, dataset, num_workers)
        all_corrupted.extend([(split_name, name, path, err) for name, path, err in corrupted])

    if all_corrupted:
        print("\nCorrupted files detected:")
        for split, name, path, err in all_corrupted:
            print(f"[{split}] {name} ({path}): {err}")
        sys.exit(1)

    print("\nNo corrupted files detected.")


if __name__ == "__main__":
    main()
