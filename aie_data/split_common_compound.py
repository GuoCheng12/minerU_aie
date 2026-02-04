#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

KEEP_FIELDS = [
    "id",
    "code",
    "SMILES",
    "reference",
    "molecular_weight",
    "emission_solid",
    "emission_aggr",
    "features_id",
    "mechanism_id",
]
KEEP_FIELDS_RAG = [*KEEP_FIELDS, "doi"]

_PLACEHOLDER_CODES = {"/", "-", "--", "---", "_"}
_NUMBER_LABEL_RE = re.compile(r"^(?:\([RS]\)-)?\d+[A-Za-z]?$")
_LETTER_RE = re.compile(r"[A-Za-z\u0370-\u03FF]")


def _has_doi(value: object | None) -> bool:
    if value is None:
        return False
    doi = str(value).strip()
    if not doi:
        return False
    return doi.lower() not in {"nan", "none", "null"}


def _is_valid_code(value: object | None) -> bool:
    if value is None:
        return False
    code = str(value).strip()
    if not code:
        return False
    if code in _PLACEHOLDER_CODES:
        return False
    if any(ch.isspace() for ch in code):
        return False
    if _NUMBER_LABEL_RE.fullmatch(code):
        return False
    if not _LETTER_RE.search(code):
        return False
    if ", " in code or "，" in code or ";" in code or "；" in code:
        return False
    return True


def split_common_compound(input_csv: Path, test_csv: Path, rag_csv: Path) -> None:
    test_count = rag_count = dropped_count = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f_in, test_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f_test, rag_csv.open("w", encoding="utf-8", newline="") as f_rag:
        reader = csv.DictReader(f_in)
        test_writer = csv.DictWriter(f_test, fieldnames=KEEP_FIELDS, extrasaction="ignore")
        rag_writer = csv.DictWriter(
            f_rag, fieldnames=KEEP_FIELDS_RAG, extrasaction="ignore"
        )
        test_writer.writeheader()
        rag_writer.writeheader()

        for row in reader:
            if (row.get("features_id") or "").strip() != "AIE":
                dropped_count += 1
                continue
            if not _is_valid_code(row.get("code")):
                dropped_count += 1
                continue

            if _has_doi(row.get("doi")):
                out_row = {k: (row.get(k) or "") for k in KEEP_FIELDS_RAG}
                out_row["id"] = out_row["id"].strip()
                out_row["code"] = out_row["code"].strip()
                out_row["doi"] = out_row["doi"].strip()
                rag_writer.writerow(out_row)
                rag_count += 1
            else:
                out_row = {k: (row.get(k) or "") for k in KEEP_FIELDS}
                out_row["id"] = out_row["id"].strip()
                out_row["code"] = out_row["code"].strip()
                test_writer.writerow(out_row)
                test_count += 1

    print(
        f"Done. test={test_count} rows, rag={rag_count} rows, dropped={dropped_count} rows."
    )
    print(f"Wrote: {test_csv}")
    print(f"Wrote: {rag_csv}")


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_input = here / "common_compound.csv"
    default_test = here / "test_compound.csv"
    default_rag = here / "rag_compound.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Split aie_data/common_compound.csv into test_compound.csv (no DOI) and "
            "rag_compound.csv (has DOI), while filtering columns/rows."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--test-out", type=Path, default=default_test)
    parser.add_argument("--rag-out", type=Path, default=default_rag)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_common_compound(args.input, args.test_out, args.rag_out)


if __name__ == "__main__":
    main()
