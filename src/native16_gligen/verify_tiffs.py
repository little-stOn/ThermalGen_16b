"""Validate that a Native16 generation directory contains one TIFF per request."""

from __future__ import annotations

import argparse
from pathlib import Path

from native16_gligen.data import load_generation_requests
from native16_gligen.evaluation import require_expected_tiffs
from native16_gligen.requests import sanitize_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = [
        f"{sanitize_filename(str(request['name']))}.tiff"
        for request in load_generation_requests(args.requests)
    ]
    if len(expected) != len(set(expected)):
        raise ValueError("Request names collide after TIFF filename sanitization")
    require_expected_tiffs(args.image_dir, expected)
    print(f"verified_tiffs={len(expected)} image_dir={args.image_dir}")


if __name__ == "__main__":
    main()
