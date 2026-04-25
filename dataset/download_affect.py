import argparse
import os

import gdown


DATASETS = {
    "mosi": {
        "path": "data/mosi/mosi_data.pkl",
        "file_id": "1_XdzdW8UNG1TTS6QcX10uhoS6N11OBit",
    },
    "mosei": {
        "path": "data/mosei/mosei_senti_data.pkl",
        "file_id": "180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz",
    },
    "humor": {
        "path": "data/humor/humor.pkl",
        "file_id": "1L5slPmYyhEVtwGyM1kgcFMjeBpXLZGT0",
    },
    "sarcasm": {
        "path": "data/sarcasm/sarcasm.pkl",
        "file_id": "1EMBUmUL5B0PTncGx3L-sBElGOmjFBR_h",
    },
}


def download_dataset(name, overwrite=False):
    spec = DATASETS[name]
    path = spec["path"]
    if os.path.exists(path) and not overwrite:
        print(f"{name}: found {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={spec['file_id']}"
    print(f"{name}: downloading to {path}")
    gdown.download(url, path, quiet=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    for dataset in args.datasets:
        download_dataset(dataset, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
