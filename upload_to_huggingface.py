"""Upload trained zen-router artifacts to Hugging Face (zenlm/zen-router)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zenlm/zen-router")
    ap.add_argument("--path", type=Path, default=Path("out/zen-router"))
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(args.path), repo_id=args.repo)
    api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=args.repo)
    print(f"uploaded {args.path} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
