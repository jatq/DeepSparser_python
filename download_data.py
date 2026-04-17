"""
Download datasets for DeepSparser experiments.

The data is hosted on Gitee. If automatic download fails due to HTTP
redirection (status 307), download manually from:
    https://gitee.com/jatq33/data/tree/master/dataset/

Usage:
    python download_data.py
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REMOTE_ROOT = "https://gitee.com/jatq33/data/raw/master/dataset/"
MANUAL_URL = "https://gitee.com/jatq33/data/tree/master/dataset/"
LOCAL_ROOT = Path("dataset")

FILES = [
    "real/high_snr.npy",
    "real/low_snr.npy",
    "real/noises.npy",
    "synthetic/training_data.npz",
    "synthetic/testing_data.npz",
]


def download():
    for file in FILES:
        local_path = LOCAL_ROOT / file
        if local_path.exists():
            logger.info("Already exists, skipping: %s", local_path)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = REMOTE_ROOT + file
        logger.info("Downloading %s ...", url)

        try:
            resp = requests.get(url, allow_redirects=False, timeout=60)
        except requests.RequestException as e:
            logger.error("Network error for %s: %s", file, e)
            continue

        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            size_mb = len(resp.content) / (1024 * 1024)
            logger.info("Saved %s (%.1f MB)", local_path, size_mb)
        elif resp.status_code in (301, 302, 307):
            logger.warning(
                "Redirect detected for %s. Please download manually from:\n  %s\n"
                "and place the file at: %s",
                file, MANUAL_URL, local_path,
            )
        else:
            logger.error("HTTP %d for %s", resp.status_code, file)


if __name__ == "__main__":
    download()
