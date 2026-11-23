#!/usr/bin/env python3
"""Build / merge a PEP 503 simple index for fplt-mc on GitHub Pages.

Layout written under ``--out`` (typically a checkout of ``gh-pages``)::

    index.html
    simple/fplt-mc/index.html
    wheels/<wheel files>

Existing wheels under ``out/wheels/`` are kept. New wheels from ``--wheels``
are copied in. Publishing the same filename twice with different bytes is an
error (immutable versions). No credentials — the Pages site is public.

Usage::

    python scripts/mc_simple_index.py --wheels dist --out site
"""

from __future__ import annotations

import argparse
import hashlib
import html
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_root(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(
        """<!DOCTYPE html>
<html><head><title>fplt-mc wheel index</title></head>
<body>
<h1>fplt-mc</h1>
<p>Public PEP 503 simple index for the <code>fplt-mc</code> native kernel.</p>
<ul>
  <li><a href="simple/fplt-mc/">simple/fplt-mc/</a></li>
</ul>
<p>Configure uv with <code>url = "https://e-mcm.github.io/fplt/simple/"</code>
(explicit index). Never overwrite a published wheel filename.</p>
</body></html>
""",
        encoding="utf-8",
    )


def _merge_wheels(wheels_dir: Path, dest_wheels: Path) -> list[Path]:
    dest_wheels.mkdir(parents=True, exist_ok=True)
    for src in sorted(wheels_dir.glob("*.whl")):
        dest = dest_wheels / src.name
        if dest.exists():
            if _sha256(dest) != _sha256(src):
                raise SystemExit(
                    f"refusing to overwrite {dest.name}: content differs "
                    "(wheels are immutable; bump fplt-mc version)"
                )
            continue
        shutil.copy2(src, dest)
    return sorted(dest_wheels.glob("*.whl"))


def _write_simple(out: Path, wheels: list[Path]) -> None:
    simple = out / "simple" / "fplt-mc"
    simple.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!DOCTYPE html>",
        "<html><head><title>Links for fplt-mc</title></head>",
        "<body><h1>Links for fplt-mc</h1>",
    ]
    for whl in wheels:
        digest = _sha256(whl)
        href = f"../../wheels/{html.escape(whl.name)}#sha256={digest}"
        lines.append(f'<a href="{href}">{html.escape(whl.name)}</a><br/>')
    lines.append("</body></html>\n")
    (simple / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Merge wheels into a public simple index tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheels",
        type=Path,
        required=True,
        help="Directory containing new .whl files to publish",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output site root (gh-pages tree)",
    )
    args = parser.parse_args(argv)

    if not args.wheels.is_dir():
        print(f"wheels dir missing: {args.wheels}", file=sys.stderr)
        return 1

    _write_root(args.out)
    all_wheels = _merge_wheels(args.wheels, args.out / "wheels")
    if not all_wheels:
        print("no wheels to index", file=sys.stderr)
        return 1
    _write_simple(args.out, all_wheels)
    print(f"indexed {len(all_wheels)} wheel(s) under {args.out}")
    for w in all_wheels:
        print(f"  {w.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
