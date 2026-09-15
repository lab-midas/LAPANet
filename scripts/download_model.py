"""
download_model.py

Downloads the model checkpoint from the Hugging Face Hub.

Usage:
    python download_model.py
    python download_model.py --output-dir ./checkpoints
"""
from pathlib import Path
import sys
# Resolve project root dynamically based on script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent#.parent
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from config.hf_config import HF_REPO_ID, CHECKPOINT_NAME


def download_model(
    repo_id: HF_REPO_ID,
    checkpoint_filename = CHECKPOINT_NAME,
    output_dir = None,
    token = None,
) -> str:
    """
    Download a single checkpoint file from a Hugging Face model repository.

    Args:
        repo_id: Hugging Face repository id (e.g. "AyaGhoul/LAPANet").
        checkpoint_filename: File name of the checkpoint inside the repo.
        output_dir: Optional local directory to store the file.
                    If None, the HF cache directory is used.
        token: Optional Hugging Face auth token (for private repos).
               Falls back to the HF_TOKEN environment variable.

    Returns:
        Absolute local path to the downloaded checkpoint.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    print(f"[download_model] repo_id           = {repo_id}")
    print(f"[download_model] checkpoint        = {checkpoint_filename}")
    print(f"[download_model] local_dir         = {output_dir or '(HF cache)'}")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
        local_dir=output_dir,
        token=token,
    )

    print(f"[download_model] downloaded to     = {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LAPANet checkpoint from HF Hub.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the checkpoint. Defaults to the HF cache.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF auth token for private repos (defaults to $HF_TOKEN).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        model_path = download_model(
            repo_id=HF_REPO_ID,
            checkpoint_filename=CHECKPOINT_NAME,
            output_dir=args.output_dir,
            token=args.token,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[download_model] ERROR: {exc}", file=sys.stderr)
        return 1

    # Print just the path so it can be captured by other scripts if needed.
    print(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())