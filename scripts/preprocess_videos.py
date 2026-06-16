from __future__ import annotations

import argparse
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.preprocessing.video import preprocess_directory
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample raw videos to target fps/size")
    parser.add_argument("--src", required=True, help="Source directory with raw .mp4 files")
    parser.add_argument("--dst", required=True, help="Destination directory")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--frame-size", type=int, default=None)
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("preprocess")

    cfg = to_dict(load_config())["features"]["preprocess"]
    target_fps = args.target_fps or int(cfg["target_fps"])
    frame_size = args.frame_size or int(cfg["frame_size"])

    n = preprocess_directory(
        Path(args.src),
        Path(args.dst),
        target_fps=target_fps,
        frame_size=frame_size,
    )
    log.info("preprocess.done", processed=n, src=args.src, dst=args.dst)


if __name__ == "__main__":
    main()
