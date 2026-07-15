#!/usr/bin/env python3
"""Build one PDF per book directory with pandoc."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_CJK_FONT = "Noto Serif CJK SC"
DEFAULT_MAIN_FONT = "Noto Serif CJK SC"
DEFAULT_MONO_FONT = "Noto Sans Mono CJK SC"


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_book_dirs(root: Path) -> list[Path]:
    book_dirs = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("book-"):
            continue
        if collect_markdown_files(path):
            book_dirs.append(path)
    return sorted(book_dirs, key=natural_key)


def collect_markdown_files(book_dir: Path) -> list[Path]:
    files: list[Path] = []

    for name in ("简介.md", "目录.md"):
        path = book_dir / name
        if path.exists():
            files.append(path)

    chapters_dir = book_dir / "chapters"
    if chapters_dir.exists():
        files.extend(sorted(chapters_dir.glob("*.md"), key=natural_key))
    else:
        chapter_files = [
            path
            for path in book_dir.glob("*.md")
            if path.name not in {"简介.md", "目录.md"}
        ]
        files.extend(sorted(chapter_files, key=natural_key))

    return files


def relative_to_book(book_dir: Path, files: list[Path]) -> list[str]:
    return [str(path.relative_to(book_dir)) for path in files]


def default_pdf_engine_opts(engine: str) -> list[str]:
    engine_name = Path(engine).name
    if engine_name in {"xelatex", "lualatex", "pdflatex"}:
        return ["-interaction=nonstopmode", "-halt-on-error"]
    return []


def build_pdf(
    pandoc: str,
    engine: str,
    engine_opts: list[str],
    book_dir: Path,
    files: list[Path],
    output_name: str,
    main_font: str,
    cjk_font: str,
    mono_font: str,
    dry_run: bool,
) -> None:
    output_path = book_dir / output_name.format(book=book_dir.name)
    command = [
        pandoc,
        *relative_to_book(book_dir, files),
        "--from",
        "markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables+fenced_code_blocks",
        "--pdf-engine",
        engine,
    ]
    for opt in engine_opts:
        command.extend(["--pdf-engine-opt", opt])

    command.extend(
        [
        "-V",
        "documentclass=article",
        "-V",
        f"mainfont={main_font}",
        "-V",
        f"CJKmainfont={cjk_font}",
        "-V",
        f"monofont={mono_font}",
        "-V",
        "geometry:margin=2.2cm",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=blue",
        "-V",
        "urlcolor=blue",
        "--toc",
        "--toc-depth",
        "3",
        "-o",
        output_path.name,
        ]
    )

    if dry_run:
        print(f"[dry-run] ({book_dir}) {' '.join(command)}")
        return

    print(f"[build] {book_dir.name} -> {output_path}")
    subprocess.run(command, cwd=book_dir, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDFs for book-* directories and place them inside each book directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--book",
        action="append",
        help="Build only this book directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-name",
        default="{book}.pdf",
        help="Output file name template. {book} is replaced with the book directory name.",
    )
    parser.add_argument(
        "--pandoc",
        default="pandoc",
        help="pandoc executable name or path.",
    )
    parser.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="PDF engine passed to pandoc. Defaults to xelatex.",
    )
    parser.add_argument(
        "--pdf-engine-opt",
        action="append",
        help=(
            "Option passed to the PDF engine. Can be passed multiple times. "
            "Defaults to non-interactive xelatex/lualatex/pdflatex options."
        ),
    )
    parser.add_argument(
        "--main-font",
        default=DEFAULT_MAIN_FONT,
        help=f"Latin main font. Defaults to {DEFAULT_MAIN_FONT}.",
    )
    parser.add_argument(
        "--cjk-font",
        default=DEFAULT_CJK_FONT,
        help=f"CJK main font. Defaults to {DEFAULT_CJK_FONT}.",
    )
    parser.add_argument(
        "--mono-font",
        default=DEFAULT_MONO_FONT,
        help=f"Monospace font. Defaults to {DEFAULT_MONO_FONT}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pandoc commands without generating PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        print(f"Repository root does not exist: {root}", file=sys.stderr)
        return 2

    if shutil.which(args.pandoc) is None:
        print(
            f"Cannot find {args.pandoc!r}. Install pandoc first, or pass --pandoc /path/to/pandoc.",
            file=sys.stderr,
        )
        return 2

    if shutil.which(args.pdf_engine) is None:
        print(
            f"Cannot find PDF engine {args.pdf_engine!r}. Install it first, or pass --pdf-engine.",
            file=sys.stderr,
        )
        return 2

    if args.book:
        book_dirs = [(root / book).resolve() for book in args.book]
    else:
        book_dirs = discover_book_dirs(root)

    if not book_dirs:
        print("No book directories found.", file=sys.stderr)
        return 1

    failures: list[tuple[Path, Exception]] = []
    for book_dir in book_dirs:
        if not book_dir.exists() or not book_dir.is_dir():
            failures.append((book_dir, FileNotFoundError(book_dir)))
            continue

        files = collect_markdown_files(book_dir)
        if not files:
            failures.append((book_dir, RuntimeError("no markdown files found")))
            continue

        try:
            build_pdf(
                pandoc=args.pandoc,
                engine=args.pdf_engine,
                engine_opts=(
                    args.pdf_engine_opt
                    if args.pdf_engine_opt is not None
                    else default_pdf_engine_opts(args.pdf_engine)
                ),
                book_dir=book_dir,
                files=files,
                output_name=args.output_name,
                main_font=args.main_font,
                cjk_font=args.cjk_font,
                mono_font=args.mono_font,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            failures.append((book_dir, exc))

    if failures:
        print("\nSome books failed:", file=sys.stderr)
        for book_dir, error in failures:
            print(f"- {book_dir}: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
