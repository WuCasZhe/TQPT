from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a local ChatGLM2-6B snapshot")
    parser.add_argument("--repo-id", default="THUDM/chatglm2-6b")
    parser.add_argument("--revision", default="v1.0")
    parser.add_argument("--output-dir", type=Path, default=Path("models/chatglm2-6b"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    from huggingface_hub import snapshot_download

    output_dir = args.output_dir.expanduser().resolve()
    print(
        f"Connecting to {args.repo_id}@{args.revision}; "
        "download progress will appear after the file list is loaded...",
        flush=True,
    )
    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the same command again to resume.")
        raise SystemExit(130) from None
    print(f"ChatGLM2-6B downloaded to: {path}")


if __name__ == "__main__":
    main()
