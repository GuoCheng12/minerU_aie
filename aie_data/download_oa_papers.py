#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


_DOI_INVALID_VALUES = {"", "nan", "none", "null"}
_DOI_PREFIX_RE = re.compile(r"^(?:doi:\s*)", re.IGNORECASE)
_DOI_URL_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


def normalize_doi(raw: object | None) -> str | None:
    if raw is None:
        return None
    doi = str(raw).strip()
    if not doi:
        return None
    doi = _DOI_PREFIX_RE.sub("", doi).strip()
    doi = _DOI_URL_PREFIX_RE.sub("", doi).strip()
    if doi.lower() in _DOI_INVALID_VALUES:
        return None
    return doi


def _safe_filename_for_doi(doi: str, suffix: str = ".pdf") -> str:
    normalized = doi.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("_")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    base = f"{sanitized}__{digest}" if sanitized else digest
    return f"{base}{suffix}"


def iter_dois_from_csv(csv_path: Path) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        field_map = {name.lower(): name for name in reader.fieldnames}
        doi_field = field_map.get("doi")
        if not doi_field:
            raise ValueError(f"No DOI column found in {csv_path} (expected header 'doi').")

        for row in reader:
            doi = normalize_doi(row.get(doi_field))
            if doi:
                yield doi


@dataclass(frozen=True)
class LookupResult:
    doi: str
    pdf_url: str | None
    is_oa: bool | None
    evidence: str | None


def unpaywall_lookup(user_agent: str, doi: str, email: str) -> LookupResult:
    # Unpaywall API docs: https://unpaywall.org/products/api
    doi_path = quote(doi, safe="/")
    url = f"https://api.unpaywall.org/v2/{doi_path}?{urlencode({'email': email})}"
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    is_oa = data.get("is_oa")
    evidence = data.get("oa_status") or data.get("evidence") or None

    best = data.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf")
    if not pdf_url:
        for loc in data.get("oa_locations") or []:
            if isinstance(loc, dict) and loc.get("url_for_pdf"):
                pdf_url = loc.get("url_for_pdf")
                break

    if not pdf_url:
        landing = best.get("url")
        if isinstance(landing, str) and landing.lower().endswith(".pdf"):
            pdf_url = landing

    return LookupResult(doi=doi, pdf_url=pdf_url, is_oa=is_oa, evidence=evidence)


def download_pdf(user_agent: str, pdf_url: str, out_path: Path, *, dry_run: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    req = Request(pdf_url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=60) as resp:
        content_type = (resp.headers.get("content-type") or "").lower()
        first_chunk = resp.read(1024 * 256)
        if not first_chunk.startswith(b"%PDF-") and "pdf" not in content_type:
            raise ValueError(f"Not a PDF (content-type={content_type!r})")

        with tmp_path.open("wb") as f:
            if first_chunk:
                f.write(first_chunk)
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)

    tmp_path.replace(out_path)


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_csv = here / "rag_compound.csv"
    default_out_dir = here / "papers_oa"

    parser = argparse.ArgumentParser(
        description=(
            "Legally download Open Access PDFs by DOI from rag_compound.csv using Unpaywall."
        )
    )
    parser.add_argument("--input", type=Path, default=default_csv, help="Path to rag_compound.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory to save PDFs and report CSV",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("UNPAYWALL_EMAIL", ""),
        help="Email for Unpaywall API (or set UNPAYWALL_EMAIL env var)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between DOI lookups (seconds) to be polite to the API",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max DOIs to process (0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve OA PDF URLs and write report, do not download files",
    )
    parser.add_argument(
        "--list-dois",
        action="store_true",
        help="Only list deduplicated DOIs and exit (no network calls)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_csv: Path = args.input

    user_agent = "MinerU-AIE-DOI-Downloader/1.0 (+https://github.com/opendatalab/MinerU)"

    seen: set[str] = set()
    dois: list[str] = []
    for doi in iter_dois_from_csv(input_csv):
        if doi in seen:
            continue
        seen.add(doi)
        dois.append(doi)
        if args.max and len(dois) >= args.max:
            break

    if args.list_dois:
        for doi in dois:
            print(doi)
        print(f"Total unique DOIs: {len(dois)}")
        return

    if not args.email.strip():
        raise SystemExit(
            "Missing --email (required by Unpaywall). You can also set UNPAYWALL_EMAIL."
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "download_report.csv"

    stats = {
        "downloaded": 0,
        "skipped_exists": 0,
        "no_open_access_pdf": 0,
        "lookup_failed": 0,
        "download_failed": 0,
    }

    with report_path.open("w", encoding="utf-8", newline="") as f_report:
        writer = csv.DictWriter(
            f_report,
            fieldnames=[
                "doi",
                "status",
                "is_oa",
                "evidence",
                "pdf_url",
                "file_path",
                "error",
            ],
        )
        writer.writeheader()

        for i, doi in enumerate(dois, start=1):
            print(f"[{i}/{len(dois)}] {doi}")
            try:
                lookup = unpaywall_lookup(user_agent, doi, args.email)
            except (HTTPError, URLError, json.JSONDecodeError, TimeoutError, ValueError) as e:
                stats["lookup_failed"] += 1
                writer.writerow(
                    {
                        "doi": doi,
                        "status": "lookup_failed",
                        "is_oa": "",
                        "evidence": "",
                        "pdf_url": "",
                        "file_path": "",
                        "error": str(e),
                    }
                )
                time.sleep(max(0.0, float(args.delay)))
                continue

            if not lookup.pdf_url:
                stats["no_open_access_pdf"] += 1
                writer.writerow(
                    {
                        "doi": doi,
                        "status": "no_open_access_pdf",
                        "is_oa": lookup.is_oa,
                        "evidence": lookup.evidence or "",
                        "pdf_url": "",
                        "file_path": "",
                        "error": "",
                    }
                )
                time.sleep(max(0.0, float(args.delay)))
                continue

            out_path = out_dir / _safe_filename_for_doi(doi)
            if out_path.exists() and out_path.stat().st_size > 0 and not args.dry_run:
                stats["skipped_exists"] += 1
                writer.writerow(
                    {
                        "doi": doi,
                        "status": "skipped_exists",
                        "is_oa": lookup.is_oa,
                        "evidence": lookup.evidence or "",
                        "pdf_url": lookup.pdf_url,
                        "file_path": str(out_path),
                        "error": "",
                    }
                )
                time.sleep(max(0.0, float(args.delay)))
                continue

            try:
                download_pdf(user_agent, lookup.pdf_url, out_path, dry_run=args.dry_run)
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as e:
                stats["download_failed"] += 1
                writer.writerow(
                    {
                        "doi": doi,
                        "status": "download_failed",
                        "is_oa": lookup.is_oa,
                        "evidence": lookup.evidence or "",
                        "pdf_url": lookup.pdf_url,
                        "file_path": str(out_path),
                        "error": str(e),
                    }
                )
                time.sleep(max(0.0, float(args.delay)))
                continue

            stats["downloaded"] += 1
            writer.writerow(
                {
                    "doi": doi,
                    "status": "downloaded" if not args.dry_run else "resolved_only",
                    "is_oa": lookup.is_oa,
                    "evidence": lookup.evidence or "",
                    "pdf_url": lookup.pdf_url,
                    "file_path": str(out_path),
                    "error": "",
                }
            )
            time.sleep(max(0.0, float(args.delay)))

    print(f"Processed {len(dois)} DOIs")
    print(f"Report: {report_path}")
    for k, v in stats.items():
        print(f"{k}: {v}")
    if args.dry_run:
        print("Dry-run enabled: no files were downloaded.")


if __name__ == "__main__":
    main()
