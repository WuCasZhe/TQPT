from __future__ import annotations

import argparse
from typing import Optional, Sequence


COMMANDS = {
    "prepare-data": "tqpt.data",
    "build-tokenizer": "tqpt.tokenizer",
    "train-qlora": "tqpt.llamafactory_stage",
    "train-prefix": "tqpt.prefix_train",
    "infer": "tqpt.inference",
    "experiments": "tqpt.experiments",
    "pipeline": "tqpt.pipeline",
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="TQPT command dispatcher")
    parser.add_argument("command", choices=COMMANDS)
    args, remainder = parser.parse_known_args(argv)
    module_name = COMMANDS[args.command]
    module = __import__(module_name, fromlist=["main"])
    module.main(remainder)


if __name__ == "__main__":
    main()
