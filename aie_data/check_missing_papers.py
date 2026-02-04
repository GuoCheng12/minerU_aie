#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _sort_key_id(value: str) -> tuple[int, str]:
    s = (value or "").strip()
    try:
        return (0, f"{int(s):012d}")
    except Exception:
        return (1, s)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"[error] Missing CSV header: {csv_path}")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        id_field = field_map.get("id")
        if not id_field:
            raise SystemExit(f"[error] CSV missing 'id' column: {csv_path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            row = {k: (v if v is not None else "") for k, v in row.items()}
            row["id"] = (row.get(id_field) or "").strip()
            rows.append(row)
        return rows


def _collect_paper_ids(papers_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in papers_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".pdf":
            continue
        ids.add(p.stem.strip())
    return ids


def main() -> int:
    here = Path(__file__).resolve().parent
    default_csv = here / "ready_data" / "rag_compound.csv"
    default_papers = here / "Papers"
    default_out = here / "ready_data" / "missing_papers.csv"

    ap = argparse.ArgumentParser(description="Check missing PDFs in aie_data/Papers for ids in rag_compound.csv")
    ap.add_argument("--csv", type=Path, default=default_csv, help="CSV file containing 'id' column")
    ap.add_argument("--papers-dir", type=Path, default=default_papers, help="Directory containing {id}.pdf files")
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Write missing rows to this CSV (id, code, doi, reference).",
    )
    ap.add_argument("--print-all", action="store_true", help="Print all missing IDs to stdout")
    args = ap.parse_args()

    csv_path: Path = args.csv
    papers_dir: Path = args.papers_dir
    out_path: Path = args.out

    if not csv_path.exists():
        raise SystemExit(f"[error] CSV not found: {csv_path}")
    if not papers_dir.exists():
        raise SystemExit(f"[error] Papers dir not found: {papers_dir}")

    rows = _read_csv_rows(csv_path)
    csv_ids = [r.get("id", "").strip() for r in rows if (r.get("id", "").strip())]
    csv_id_set = set(csv_ids)

    paper_id_set = _collect_paper_ids(papers_dir)

    missing_ids = sorted((csv_id_set - paper_id_set), key=_sort_key_id)
    missing_rows: list[dict[str, str]] = []
    missing_lookup = set(missing_ids)
    for r in rows:
        rid = (r.get("id") or "").strip()
        if rid in missing_lookup:
            missing_rows.append(
                {
                    "id": rid,
                    "code": (r.get("code") or "").strip(),
                    "doi": (r.get("doi") or "").strip(),
                    "reference": (r.get("reference") or "").strip(),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "code", "doi", "reference"])
        w.writeheader()
        w.writerows(missing_rows)

    print(f"[check] csv_rows={len(rows)} csv_unique_ids={len(csv_id_set)}")
    print(f"[check] papers_dir={papers_dir} pdf_files={len(list(papers_dir.glob('*.pdf')))+len(list(papers_dir.glob('*.PDF')))} paper_unique_ids={len(paper_id_set)}")
    print(f"[check] missing_count={len(missing_ids)}")
    print(f"[check] missing_report={out_path}")

    if missing_ids:
        preview = missing_ids if args.print_all else missing_ids[:50]
        print("[check] missing_ids:")
        for mid in preview:
            print(mid)
        if not args.print_all and len(missing_ids) > len(preview):
            print(f"[check] ... ({len(missing_ids) - len(preview)} more, see report CSV)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

