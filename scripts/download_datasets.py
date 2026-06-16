from __future__ import annotations

import argparse
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.ingestion.downloaders import get_downloader
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a sign-language dataset")
    parser.add_argument("--dataset", required=True, choices=["wlasl", "msasl", "asl_citizen"])
    parser.add_argument("--root", default=None, help="Override raw root directory")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("download")

    cfg = load_config([f"data={args.dataset}"])
    data_cfg = to_dict(cfg)["data"]
    root = Path(args.root) if args.root else Path(to_dict(cfg)["paths"]["raw_dir"]) / args.dataset
    dl = get_downloader(args.dataset, root=root, url=data_cfg["url"])
    if dl.is_complete() and not args.force:
        log.info("download.skip", dataset=args.dataset, root=str(root))
        return
    dl.download(force=args.force)


if __name__ == "__main__":
    main()
