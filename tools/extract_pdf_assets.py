from __future__ import annotations

import argparse
from pathlib import Path

import pypdf

DESCRIPTION = (
    "Dump embedded raster images from the CALM preprint PDF; the preprint ships "
    "its equations and Tables 2-4 as images, not as extractable text."
)


def dump_page(page, page_no: int, outdir: Path) -> None:
    for idx, image in enumerate(page.images):
        suffix = Path(image.name).suffix or ".png"
        out = outdir / f"p{page_no:02d}_{idx}{suffix}"
        out.write_bytes(image.data)
        print(f"{out}  {len(image.data)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--pages", default="", help="1-based comma list; empty means all")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.pages.split(",") if x.strip()} if args.pages else None
    reader = pypdf.PdfReader(str(args.pdf))

    pages = list(enumerate(reader.pages, start=1))
    selected = [(n, p) for n, p in pages if wanted is None or n in wanted]
    for page_no, page in selected:
        dump_page(page, page_no, args.outdir)


if __name__ == "__main__":
    main()
