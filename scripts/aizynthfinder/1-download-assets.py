"""Module with script to download public data

Usage:

 uv run scripts/aizynthfinder/1-download-assets.py data/0-assets/model-configs/aizynthfinder
"""

import argparse
import os
import sys
from pathlib import Path

import requests
import tqdm

FILES_TO_DOWNLOAD = {
    "policy_model_onnx": {
        "filename": "uspto_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
    },
    "template_file": {
        "filename": "uspto_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    },
    "ringbreaker_model_onnx": {
        "filename": "uspto_ringbreaker_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    },
    "ringbreaker_templates": {
        "filename": "uspto_ringbreaker_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    },
    "stock": {
        "filename": "zinc_stock.hdf5",
        "url": "https://ndownloader.figshare.com/files/23086469",
    },
    "filter_policy_onnx": {
        "filename": "uspto_filter_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    },
    "retrostar_value_model": {
        "filename": "retrostar_value_model.pickle",
        "url": "https://github.com/MolecularAI/PaRoutes/blob/main/publication/retrostar_value_model.pickle?raw=true",
    },
}

YAML_TEMPLATE = """expansion:
  uspto:
    - {}
    - {}
  ringbreaker:
    - {}
    - {}
filter:
  uspto: {}
stock:
  zinc: {}
"""


def _download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True)
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    """Entry-point for CLI"""
    parser = argparse.ArgumentParser("download_public_data")
    parser.add_argument(
        "path",
        type=Path,
        default=".",
        help="the path to download the files",
    )
    path = parser.parse_args().path
    path.mkdir(parents=True, exist_ok=True)

    try:
        for filespec in FILES_TO_DOWNLOAD.values():
            _download_file(filespec["url"], os.path.join(path, filespec["filename"]))
    except requests.HTTPError as err:
        print(f"Download failed with message {str(err)}")
        sys.exit(1)

    with open(os.path.join(path, "config.yml"), "w") as fileobj:
        path = os.path.abspath(path)
        fileobj.write(
            YAML_TEMPLATE.format(
                os.path.join(path, FILES_TO_DOWNLOAD["policy_model_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["template_file"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["ringbreaker_model_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["ringbreaker_templates"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["filter_policy_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["stock"]["filename"]),
            )
        )
    print("Configuration file written to config.yml")


if __name__ == "__main__":
    main()
