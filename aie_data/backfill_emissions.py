#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
_VENV_PY_UNIX = REPO_ROOT / ".venv" / "bin" / "python"
_VENV_PY_WIN = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _preferred_python() -> str:
    for p in (_VENV_PY_UNIX, _VENV_PY_WIN):
        if p.exists() and p.is_file():
            return str(p)
    return sys.executable

READY_FIELDS = [
    "id",
    "code",
    "SMILES",
    "reference",
    "molecular_weight",
    "features_id",
    "mechanism_id",
    "doi",
]

OUTPUT_FIELDS = [
    "id",
    "code",
    "SMILES",
    "reference",
    "molecular_weight",
    "emission_solid",
    "emission_aggr",
    "features_id",
    "mechanism_id",
    "doi",
]


class BackfillError(RuntimeError):
    pass


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BackfillError(f"File not found: {path}") from exc
    except Exception as exc:
        raise BackfillError(f"Failed to read file: {path}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _backup_file(path: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _fmt_duration_s(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def _log_progress(i: int, total: int, paper_id: str, *, step: str, detail: str = "") -> None:
    prefix = f"[run] [{i}/{total}] id={paper_id} step={step}"
    msg = f"{prefix} {detail}".rstrip()
    print(msg)


def _prepare_csv(*, input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise BackfillError(f"Missing CSV header: {input_csv}")
        writer = csv.DictWriter(f_out, fieldnames=READY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        count = 0
        for row in reader:
            out_row = {k: (row.get(k) if row.get(k) is not None else "") for k in READY_FIELDS}
            writer.writerow(out_row)
            count += 1
    print(f"[prepare] wrote {count} rows -> {output_csv}")


def _iter_content_items(items: Any) -> Iterable[dict[str, Any]]:
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                yield item


def _render_inline(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    content = item.get("content", "")
    if not isinstance(content, str):
        return ""
    if item_type in {"equation_inline", "equation"}:
        return f"${content}$"
    return content


def _html_table_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except Exception:
        text = re.sub(r"<\s*br\s*/?\s*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</\s*tr\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</\s*t[dh]\s*>", "\t", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return soup.get_text(" ", strip=True)
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _block_to_text_v2(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    content = block.get("content") or {}
    if not isinstance(content, dict):
        content = {}

    if block_type == "title":
        title_items = content.get("title_content")
        title = "".join(_render_inline(x) for x in _iter_content_items(title_items)).strip()
        level = content.get("level")
        if isinstance(level, int) and level > 0:
            prefix = "#" * min(level, 6)
            return f"{prefix} {title}".strip()
        return title

    if block_type == "paragraph":
        para_items = content.get("paragraph_content")
        return "".join(_render_inline(x) for x in _iter_content_items(para_items)).strip()

    if block_type == "list":
        list_items = content.get("list_items")
        lines: list[str] = []
        for li in _iter_content_items(list_items):
            item_content = li.get("item_content")
            text = "".join(_render_inline(x) for x in _iter_content_items(item_content)).strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines).strip()

    if block_type == "table":
        caption_items = content.get("table_caption")
        caption = "".join(_render_inline(x) for x in _iter_content_items(caption_items)).strip()
        foot_items = content.get("table_footnote")
        footnote = "".join(_render_inline(x) for x in _iter_content_items(foot_items)).strip()
        html = content.get("html", "")
        table_txt = _html_table_to_text(html) if isinstance(html, str) and html else ""
        parts = []
        if caption:
            parts.append(f"Table: {caption}")
        if table_txt:
            parts.append(table_txt)
        if footnote:
            parts.append(f"Table note: {footnote}")
        return "\n".join(parts).strip()

    if block_type == "image":
        caption_items = content.get("image_caption")
        caption = "".join(_render_inline(x) for x in _iter_content_items(caption_items)).strip()
        foot_items = content.get("image_footnote")
        footnote = "".join(_render_inline(x) for x in _iter_content_items(foot_items)).strip()
        parts = []
        if caption:
            parts.append(f"Figure: {caption}")
        if footnote:
            parts.append(f"Figure note: {footnote}")
        return "\n".join(parts).strip()

    if block_type == "equation_interline":
        math = content.get("math_content", "")
        if isinstance(math, str) and math.strip():
            return f"Equation: {math.strip()}"
        return ""

    if block_type == "page_footnote":
        items = content.get("page_footnote_content")
        return "".join(_render_inline(x) for x in _iter_content_items(items)).strip()

    return ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page_start: int | None
    page_end: int | None
    text: str


def _merge_units_to_chunks(units: list[tuple[int | None, str]], *, chunk_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    def split_long_text(text: str) -> list[str]:
        text = text.strip()
        if len(text) <= chunk_chars:
            return [text]
        overlap = min(200, max(chunk_chars // 5, 0))
        parts: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            j = min(i + chunk_chars, n)
            if j < n:
                window = text[i:j]
                cut = window.rfind(" ")
                if cut >= max(int(chunk_chars * 0.6), 1):
                    j = i + cut
            if j <= i:
                j = min(i + chunk_chars, n)
            part = text[i:j].strip()
            if part:
                parts.append(part)
            if j >= n:
                break
            i = max(j - overlap, i + 1)
        return parts

    def flush() -> None:
        nonlocal cur, page_start, page_end
        if not cur:
            return
        chunk_id = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                page_start=page_start,
                page_end=page_end,
                text="\n\n".join(cur).strip(),
            )
        )
        cur = []
        page_start = None
        page_end = None

    for page_no, text in units:
        for part in split_long_text(text):
            projected = (len("\n\n".join(cur)) + 2 + len(part)) if cur else len(part)
            if cur and projected > chunk_chars:
                flush()
            cur.append(part)
            if page_no is not None:
                page_start = page_no if page_start is None else min(page_start, page_no)
                page_end = page_no if page_end is None else max(page_end, page_no)
    flush()
    return chunks


def _load_chunks_from_content_list_v2(path: Path, *, chunk_chars: int) -> list[Chunk]:
    data = json.loads(_read_text(path))
    if not isinstance(data, list):
        raise BackfillError(f"Invalid content_list_v2 format: expected list, got {type(data)}")

    units: list[tuple[int, str]] = []
    for page_idx, page in enumerate(data):
        if not isinstance(page, list):
            continue
        page_no = page_idx + 1
        for block in page:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"page_header", "page_footer", "page_number"}:
                continue
            text = _block_to_text_v2(block)
            if text:
                units.append((page_no, text))
    return _merge_units_to_chunks(units, chunk_chars=chunk_chars)


def _load_chunks_from_content_list_v1(path: Path, *, chunk_chars: int) -> list[Chunk]:
    data = json.loads(_read_text(path))
    if not isinstance(data, list):
        raise BackfillError(f"Invalid content_list format: expected list, got {type(data)}")

    units: list[tuple[int | None, str]] = []
    for block in data:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"page_header", "page_footer", "header", "footer", "page_number"}:
            continue
        page_idx = block.get("page_idx")
        page_no = (page_idx + 1) if isinstance(page_idx, int) else None

        text = block.get("text")
        if isinstance(text, str) and text.strip():
            units.append((page_no, text.strip()))
            continue

        if block_type == "table":
            caption_items = block.get("table_caption")
            if isinstance(caption_items, list):
                caption = " ".join(str(x) for x in caption_items if str(x).strip()).strip()
            else:
                caption = str(caption_items or "").strip()
            foot_items = block.get("table_footnote")
            if isinstance(foot_items, list):
                footnote = " ".join(str(x) for x in foot_items if str(x).strip()).strip()
            else:
                footnote = str(foot_items or "").strip()
            body = block.get("table_body")
            table_txt = _html_table_to_text(body) if isinstance(body, str) and body.strip() else ""
            parts = []
            if caption:
                parts.append(f"Table: {caption}")
            if table_txt:
                parts.append(table_txt)
            if footnote:
                parts.append(f"Table note: {footnote}")
            if parts:
                units.append((page_no, "\n".join(parts).strip()))
            continue

        if block_type == "image":
            caption_items = block.get("image_caption")
            if isinstance(caption_items, list):
                caption = " ".join(str(x) for x in caption_items if str(x).strip()).strip()
            else:
                caption = str(caption_items or "").strip()
            foot_items = block.get("image_footnote")
            if isinstance(foot_items, list):
                footnote = " ".join(str(x) for x in foot_items if str(x).strip()).strip()
            else:
                footnote = str(foot_items or "").strip()
            parts = []
            if caption:
                parts.append(f"Figure: {caption}")
            if footnote:
                parts.append(f"Figure note: {footnote}")
            if parts:
                units.append((page_no, "\n".join(parts).strip()))
            continue

    return _merge_units_to_chunks(units, chunk_chars=chunk_chars)


def _load_chunks_from_md(path: Path, *, chunk_chars: int) -> list[Chunk]:
    text = _read_text(path)
    units: list[tuple[int | None, str]] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                units.append((None, "\n".join(buf).strip()))
                buf = []
            buf.append(line.strip())
            continue
        if not line.strip():
            if buf:
                units.append((None, "\n".join(buf).strip()))
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        units.append((None, "\n".join(buf).strip()))
    return _merge_units_to_chunks(units, chunk_chars=chunk_chars)


def _resolve_mineru_source(doc_dir: Path) -> tuple[Path, str]:
    v2 = sorted(doc_dir.glob("*_content_list_v2.json"))
    if v2:
        return v2[0], "content_list_v2"
    v1 = sorted(doc_dir.glob("*_content_list.json"))
    if v1:
        return v1[0], "content_list"
    mds = sorted(doc_dir.glob("*.md"))
    if mds:
        return mds[0], "md"
    raise BackfillError(f"No supported MinerU output found in: {doc_dir}")


def _load_chunks(doc_dir: Path, *, chunk_chars: int) -> list[Chunk]:
    source_path, kind = _resolve_mineru_source(doc_dir)
    if kind == "content_list_v2":
        return _load_chunks_from_content_list_v2(source_path, chunk_chars=chunk_chars)
    if kind == "content_list":
        return _load_chunks_from_content_list_v1(source_path, chunk_chars=chunk_chars)
    if kind == "md":
        return _load_chunks_from_md(source_path, chunk_chars=chunk_chars)
    raise BackfillError(f"Unsupported source kind for keyword retrieval: {kind}")


def _score_chunk(text: str, *, code: str) -> float:
    t = text.lower()
    code_l = code.lower().strip()
    score = 0.0

    if code_l and code_l in t:
        score += 6.0

    if "table:" in t or "\n|" in text or " | " in text:
        score += 1.5

    emission_kw = [
        "emission",
        "fluorescence",
        "photoluminescence",
        "luminescence",
        "pl",
        "λ",
        "lambda",
        "nm",
        "max",
    ]
    solid_kw = ["solid", "film", "powder", "crystal", "thin film", "spin-coated", "cast film", "neat film"]
    aggr_kw = [
        "aggregate",
        "aggregated",
        "aggregation",
        "nanoaggregate",
        "fw",
        "water fraction",
        "thf/water",
        "thf",
        "water",
        "poor solvent",
    ]

    if any(k in t for k in emission_kw):
        score += 3.0
    if any(k in t for k in solid_kw):
        score += 2.0
    if any(k in t for k in aggr_kw):
        score += 2.0

    if "em(" in t or "λem" in t or "lambda_em" in t:
        score += 1.0

    return score


def _select_top_chunks(
    chunks: list[Chunk], *, code: str, top_k: int, max_context_chars: int
) -> list[Chunk]:
    scored = [(ch, _score_chunk(ch.text, code=code)) for ch in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    picked: list[Chunk] = []
    total = 0
    for ch, score in scored[: max(int(top_k) * 4, 1)]:
        if score <= 0:
            continue
        extra = len(ch.text) + 200
        if picked and total + extra > max_context_chars:
            break
        picked.append(ch)
        total += extra
        if len(picked) >= max(int(top_k), 1):
            break
    return picked


def _render_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        if ch.page_start and ch.page_end:
            loc = f"pages {ch.page_start}-{ch.page_end}" if ch.page_start != ch.page_end else f"page {ch.page_start}"
        else:
            loc = "unknown pages"
        parts.append(f"[Chunk {i} | {loc}]\n{ch.text}")
    return "\n\n".join(parts).strip()


_TARGET_NAME_RE = re.compile(r"(Target common name \(if present\):\s*)'[^']*'")


def _render_prompt(template: str, *, code: str, smiles: str) -> str:
    rendered = template

    # New template: replace the example target code everywhere for consistency.
    rendered = rendered.replace("t-BPITBT-TPATPE", code)

    # Old template compatibility: replace the quoted "Target common name ..." line if present.
    rendered = _TARGET_NAME_RE.sub(lambda m: f"{m.group(1)}'{code}'", rendered, count=1)
    rendered = rendered.replace("target common name TPEAn", f"target common name {code}", 1)

    smiles_value = smiles if isinstance(smiles, str) else ""
    replacement_smiles = smiles_value.strip() or "<INSERT_IF_AVAILABLE>"
    # Replace first TARGET_SMILES placeholder (handles straight/curly quotes; normalize to straight quotes).
    rendered = re.sub(
        r'(TARGET_SMILES\s*=\s*)[“”"]\s*<INSERT_IF_AVAILABLE>\s*[“”"]',
        lambda m: f'{m.group(1)}"{replacement_smiles}"',
        rendered,
        count=1,
    )
    return rendered


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_llm_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    obj = _extract_json_object(text)
    if obj:
        data = json.loads(obj)
        if isinstance(data, dict):
            return data
    raise BackfillError("LLM output is not valid JSON object.")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_nm(value: Any, *, nm_min: float, nm_max: float) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if not _is_number(value):
        return None, "invalid_type"
    v = float(value)
    if not (nm_min <= v <= nm_max):
        return None, "out_of_range"
    return v, None


def _format_nm(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _num_appears_in_text(value: float, text: str) -> bool:
    if not text:
        return False
    candidates: list[str] = []
    if abs(value - round(value)) < 1e-9:
        candidates.append(str(int(round(value))))
    candidates.append(f"{value:.6g}")
    for s in candidates:
        # Match as a standalone number, not part of a longer number.
        pattern = re.compile(rf"(?<!\\d){re.escape(s)}(?!\\d)")
        if pattern.search(text):
            return True
        if re.search(rf"(?<!\\d){re.escape(s)}\\s*nm(?!\\w)", text, flags=re.IGNORECASE):
            return True
    return False


def _suspicious_output(parsed: dict[str, Any], *, context: str) -> list[str]:
    warnings: list[str] = []
    ev = parsed.get("evidence")
    evidence: dict[str, Any] = ev if isinstance(ev, dict) else {}

    solid = parsed.get("emission_solid_or_film_nm")
    if _is_number(solid):
        src = evidence.get("solid_or_film_source")
        if not isinstance(src, str) or not src.strip():
            warnings.append("solid:missing_evidence_source")
        if not _num_appears_in_text(float(solid), context):
            warnings.append("solid:not_in_context")

    aggr = parsed.get("emission_aggr_nm")
    if _is_number(aggr):
        src = evidence.get("aggr_source")
        if not isinstance(src, str) or not src.strip():
            warnings.append("aggr:missing_evidence_source")
        if not _num_appears_in_text(float(aggr), context):
            warnings.append("aggr:not_in_context")

    return warnings


def _needs_embedding_fallback(parsed: dict[str, Any], *, min_confidence: float) -> bool:
    match = parsed.get("identity_match")
    conf = parsed.get("identity_match_confidence")
    aggr = parsed.get("emission_aggr_nm")
    solid = parsed.get("emission_solid_or_film_nm")

    if (aggr is not None and not _is_number(aggr)) or (solid is not None and not _is_number(solid)):
        return True
    if match in {"not_found", "uncertain"}:
        return True
    if conf is None or not _is_number(conf):
        return True
    if float(conf) < float(min_confidence):
        return True
    if aggr is None and solid is None:
        return True
    return False


class OpenAIChatClient:
    def __init__(self, *, api_key: str, base_url: str | None):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:
            raise BackfillError("Missing dependency: openai. Please run in MinerU virtualenv.") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        *,
        model: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: str = "auto",
        retries: int = 2,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                kwargs: dict[str, Any] = {}
                if response_format == "json":
                    # Best-effort structured output. Some relays may not support this; we'll retry without.
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    resp = self._client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": user_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except Exception:
                    if kwargs and response_format == "json":
                        resp = self._client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": user_prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    raise BackfillError(
                        "LLM returned empty content (model may not support chat.completions or relay returned tool calls)."
                    )
                return content
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise BackfillError(f"OpenAI chat failed: {last_exc}")


class OpenAIEmbedClient:
    def __init__(self, *, api_key: str, base_url: str | None):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:
            raise BackfillError("Missing dependency: openai. Please run in MinerU virtualenv.") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def embed(self, *, model: str, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=model, input=text)
        emb = resp.data[0].embedding
        return np.asarray(emb, dtype=np.float32)


def _find_pdf(pdf_dir: Path, paper_id: str) -> Path | None:
    candidates = [
        pdf_dir / f"{paper_id}.pdf",
        pdf_dir / f"{paper_id}.PDF",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _ensure_mineru_output(
    *, paper_id: str, pdf_path: Path, output_root: Path, force: bool
) -> tuple[Path, bool]:
    doc_dir = output_root / paper_id / "hybrid_auto"
    did_run = False
    if not force and doc_dir.exists():
        try:
            _resolve_mineru_source(doc_dir)
            return doc_dir, did_run
        except Exception:
            pass

    cmd = [
        _preferred_python(),
        "-m",
        "mineru.cli.client",
        "-p",
        str(pdf_path),
        "-o",
        str(output_root),
    ]
    did_run = True
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise BackfillError(
            f"MinerU failed (exit {proc.returncode}) for {pdf_path}. stderr: {proc.stderr[-2000:]}"
        )

    try:
        _resolve_mineru_source(doc_dir)
    except Exception as exc:
        raise BackfillError(f"MinerU output missing expected files in {doc_dir}") from exc
    return doc_dir, did_run


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec)) + 1e-12
    return vec / denom


def _cosine_top_k(embeddings: np.ndarray, query: np.ndarray, *, top_k: int) -> list[int]:
    if embeddings.ndim != 2:
        raise BackfillError(f"Invalid embeddings matrix shape: {embeddings.shape}")
    if query.ndim != 1:
        raise BackfillError(f"Invalid query vector shape: {query.shape}")
    if embeddings.shape[1] != query.shape[0]:
        raise BackfillError(
            f"Embedding dim mismatch: index dim {embeddings.shape[1]} vs query dim {query.shape[0]}"
        )
    sims = embeddings @ query
    top_k = min(max(int(top_k), 1), embeddings.shape[0])
    idx = np.argpartition(-sims, kth=top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return [int(i) for i in idx.tolist()]


def _read_rag_index(index_dir: Path) -> tuple[list[Chunk], np.ndarray]:
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"
    if not chunks_path.exists() or not emb_path.exists():
        raise BackfillError(f"Missing rag index files under: {index_dir}")

    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            chunks.append(
                Chunk(
                    chunk_id=int(obj.get("chunk_id", len(chunks))),
                    page_start=obj.get("page_start") if isinstance(obj.get("page_start"), int) else None,
                    page_end=obj.get("page_end") if isinstance(obj.get("page_end"), int) else None,
                    text=str(obj.get("text", "")),
                )
            )
    emb = np.load(emb_path)
    if emb.shape[0] != len(chunks):
        raise BackfillError(f"Index mismatch: {emb.shape[0]} embeddings vs {len(chunks)} chunks in {index_dir}")
    return chunks, emb


def _run_rag_chat_build(
    *,
    doc_dir: Path,
    provider: str,
    embed_model: str,
    chunk_chars: int,
    openai_api_key: str | None,
    openai_base_url: str | None,
    rebuild: bool,
) -> None:
    rag_chat_script = REPO_ROOT / "demo" / "rag_chat.py"
    cmd = [
        _preferred_python(),
        str(rag_chat_script),
        "build",
        "--doc",
        str(doc_dir),
        "--provider",
        provider,
        "--embed-model",
        embed_model,
        "--chunk-chars",
        str(chunk_chars),
    ]
    if rebuild:
        cmd.append("--rebuild")
    if provider == "openai":
        if openai_api_key:
            cmd.extend(["--openai-api-key", openai_api_key])
        if openai_base_url:
            cmd.extend(["--openai-base-url", openai_base_url])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise BackfillError(
            f"rag_chat.py build failed (exit {proc.returncode}). stderr: {proc.stderr[-2000:]}"
        )


def _ensure_rag_index(
    *,
    doc_dir: Path,
    provider: str,
    embed_model: str,
    chunk_chars: int,
    openai_api_key: str | None,
    openai_base_url: str | None,
    rebuild: bool,
) -> Path:
    index_dir = doc_dir / ".rag_index"
    meta_path = index_dir / "meta.json"
    ok = index_dir.exists() and meta_path.exists() and (index_dir / "chunks.jsonl").exists() and (index_dir / "embeddings.npy").exists()
    if ok and not rebuild:
        try:
            meta = json.loads(_read_text(meta_path))
            if (
                meta.get("provider") == provider
                and meta.get("embed_model") == embed_model
                and int(meta.get("chunk_chars", -1)) == int(chunk_chars)
            ):
                return index_dir
        except Exception:
            pass

    _run_rag_chat_build(
        doc_dir=doc_dir,
        provider=provider,
        embed_model=embed_model,
        chunk_chars=chunk_chars,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        rebuild=rebuild or ok,
    )
    return index_dir


def _build_embedding_query(*, code: str, smiles: str) -> str:
    parts: list[str] = [
        "Extract emission peak wavelengths (nm) for a target molecule.",
        "Need: aggregated-state emission peak (aggregation/THF-water/fw/aggregate) and solid/film emission peak.",
        f"Target name: {code}",
    ]
    if smiles.strip():
        parts.append(f"Target SMILES: {smiles.strip()}")
    parts.append(
        "Look for: emission/fluorescence/photoluminescence/PL, nm, λem, solid/film/powder/crystal, aggregate/aggregation, THF/water, fw."
    )
    return "\n".join(parts).strip()


def _trim_chunks(chunks: list[Chunk], *, max_context_chars: int) -> list[Chunk]:
    out: list[Chunk] = []
    total = 0
    for ch in chunks:
        extra = len(ch.text) + 200
        if out and total + extra > max_context_chars:
            break
        out.append(ch)
        total += extra
    return out


def _dedup_chunks(primary: list[Chunk], secondary: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[str] = set()
    for ch in [*primary, *secondary]:
        digest = hashlib.sha1(ch.text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(ch)
    return out


def _retrieve_by_embedding(
    *,
    doc_dir: Path,
    provider: str,
    embed_model: str,
    chunk_chars: int,
    top_k: int,
    max_context_chars: int,
    openai_api_key: str,
    openai_base_url: str | None,
    code: str,
    smiles: str,
    rebuild: bool,
) -> list[Chunk]:
    index_dir = _ensure_rag_index(
        doc_dir=doc_dir,
        provider=provider,
        embed_model=embed_model,
        chunk_chars=chunk_chars,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        rebuild=rebuild,
    )
    chunks, embeddings = _read_rag_index(index_dir)
    query = _build_embedding_query(code=code, smiles=smiles)
    embed_client = OpenAIEmbedClient(api_key=openai_api_key, base_url=openai_base_url)
    q_vec = _l2_normalize(embed_client.embed(model=embed_model, text=query))
    idxs = _cosine_top_k(embeddings, q_vec, top_k=top_k)
    retrieved = [chunks[i] for i in idxs]
    return _trim_chunks(retrieved, max_context_chars=max_context_chars)


def _write_ready_csv_with_emissions(
    *, rows: list[dict[str, str]], output_csv: Path, backup: bool
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if backup and output_csv.exists():
        backup_path = _backup_file(output_csv)
        print(f"[run] backup -> {backup_path}")

    tmp = output_csv.with_suffix(output_csv.suffix + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: (row.get(k) if row.get(k) is not None else "") for k in OUTPUT_FIELDS}
            writer.writerow(out)
    tmp.replace(output_csv)


def _build_report(*, cache_dir: Path, report_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for p in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(_read_text(p))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rows.append(
            {
                "id": str(data.get("id", "")),
                "status": str(data.get("status", "")),
                "method": str(data.get("method", "")),
                "emission_solid": str(data.get("emission_solid", "")),
                "emission_aggr": str(data.get("emission_aggr", "")),
                "identity_match": str(data.get("identity_match", "")),
                "identity_match_confidence": str(data.get("identity_match_confidence", "")),
                "error": str(data.get("error", "")),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "status",
                "method",
                "emission_solid",
                "emission_aggr",
                "identity_match",
                "identity_match_confidence",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(report_path)


def _cmd_run(args: argparse.Namespace) -> None:
    csv_path: Path = args.csv
    pdf_dir: Path = args.pdf_dir
    output_root: Path = args.mineru_output_root
    prompt_template_path: Path = args.prompt_template

    provider: str = args.provider
    retrieval: str = args.retrieval
    if provider != "openai":
        raise BackfillError("Only provider=openai is supported in backfill_emissions.py.")
    if retrieval not in {"keyword_first", "embedding_only"}:
        raise BackfillError(f"Unsupported retrieval: {retrieval}")

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY") or ""
    openai_base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL") or None
    if not openai_api_key:
        raise BackfillError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --openai-api-key.")

    chat_model: str = args.chat_model
    embed_model: str = args.embed_model
    chunk_chars: int = int(args.chunk_chars)
    top_k: int = int(args.top_k)
    temperature: float = float(args.temperature)
    max_tokens: int = int(args.max_tokens)
    max_context_chars: int = int(args.max_context_chars)
    min_confidence: float = float(args.min_confidence)
    nm_min: float = float(args.nm_min)
    nm_max: float = float(args.nm_max)
    verify: bool = bool(getattr(args, "verify", True))
    response_format: str = str(getattr(args, "response_format", "auto"))

    resume: bool = bool(args.resume)
    force: bool = bool(args.force)
    limit: int = int(args.limit)
    mineru_force: bool = bool(args.mineru_force)
    backup: bool = bool(args.backup)

    cache_dir = csv_path.parent / "emission_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = csv_path.parent / "emission_backfill_report.csv"

    template = _read_text(prompt_template_path)
    client = OpenAIChatClient(api_key=openai_api_key, base_url=openai_base_url)

    run_cfg = {
        "provider": provider,
        "retrieval": retrieval,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "chunk_chars": chunk_chars,
        "top_k": top_k,
        "max_context_chars": max_context_chars,
        "min_confidence": min_confidence,
        "nm_min": nm_min,
        "nm_max": nm_max,
        "verify": verify,
        "response_format": response_format,
        "prompt_template_sha1": hashlib.sha1(template.encode("utf-8")).hexdigest(),
    }
    cache_key = hashlib.sha1(json.dumps(run_cfg, sort_keys=True).encode("utf-8")).hexdigest()

    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise BackfillError(f"Missing CSV header: {csv_path}")
        in_rows = list(reader)

    rows: list[dict[str, str]] = []
    for row in in_rows:
        out = {k: (row.get(k) if row.get(k) is not None else "") for k in READY_FIELDS}
        out["emission_solid"] = row.get("emission_solid", "") or ""
        out["emission_aggr"] = row.get("emission_aggr", "") or ""
        out["mechanism_id"] = row.get("mechanism_id", "") or out.get("mechanism_id", "")
        rows.append(out)

    total = min(limit, len(rows)) if limit else len(rows)
    print(
        "[run] start:"
        f" rows={len(rows)}"
        f" processing={total}"
        f" provider={provider}"
        f" retrieval={retrieval}"
        f" chat_model={chat_model}"
        f" embed_model={embed_model}"
        f" verify={'on' if verify else 'off'}"
    )
    print(f"[run] cache_key={cache_key[:12]} prompt_sha1={run_cfg['prompt_template_sha1'][:12]}")

    stats: dict[str, int] = {
        "ok": 0,
        "ok_cached": 0,
        "missing_pdf": 0,
        "mineru_failed": 0,
        "llm_failed": 0,
        "other": 0,
    }

    processed = 0
    for row_idx, row in enumerate(rows, start=1):
        if row_idx > total:
            break

        paper_id = str(row.get("id", "")).strip()
        code = str(row.get("code", ""))
        smiles = str(row.get("SMILES", ""))
        cache_path = cache_dir / f"{paper_id}.json"

        if resume and not force and cache_path.exists():
            try:
                cached = json.loads(_read_text(cache_path))
                if (
                    isinstance(cached, dict)
                    and cached.get("status") == "ok"
                    and cached.get("cache_key") == cache_key
                ):
                    row["emission_solid"] = str(cached.get("emission_solid", "") or "")
                    row["emission_aggr"] = str(cached.get("emission_aggr", "") or "")
                    processed += 1
                    stats["ok_cached"] += 1
                    _log_progress(row_idx, total, paper_id, step="cache", detail="hit status=ok -> skip")
                    continue
            except Exception:
                pass

        # We're going to (re)process this row. Clear stale emissions early so failures don't keep old values.
        row["emission_solid"] = ""
        row["emission_aggr"] = ""

        _log_progress(row_idx, total, paper_id, step="start", detail=f"code={code}")

        result: dict[str, Any] = {
            "id": paper_id,
            "code": code,
            "SMILES": smiles,
            "cache_key": cache_key,
            "status": "",
            "method": "",
            "error": "",
            "identity_match": "",
            "identity_match_confidence": "",
            "emission_solid": "",
            "emission_aggr": "",
            "raw_response": "",
            "raw_response_fallback": "",
            "warnings": [],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }

        pdf_path = _find_pdf(pdf_dir, paper_id)
        if not pdf_path:
            result["status"] = "missing_pdf"
            result["error"] = f"PDF not found: {pdf_dir}/{paper_id}.pdf"
            _atomic_write_json(cache_path, result)
            processed += 1
            stats["missing_pdf"] += 1
            _log_progress(row_idx, total, paper_id, step="pdf", detail="missing -> skip")
            continue

        try:
            _log_progress(row_idx, total, paper_id, step="mineru", detail="checking outputs")
            t0 = time.time()
            doc_dir, did_run = _ensure_mineru_output(
                paper_id=paper_id, pdf_path=pdf_path, output_root=output_root, force=mineru_force
            )
            dt_s = _fmt_duration_s(time.time() - t0)
            _log_progress(
                row_idx,
                total,
                paper_id,
                step="mineru",
                detail=("parsed" if did_run else "reused") + f" ({dt_s})",
            )
        except Exception as exc:
            result["status"] = "mineru_failed"
            result["error"] = str(exc)
            _atomic_write_json(cache_path, result)
            processed += 1
            stats["mineru_failed"] += 1
            _log_progress(row_idx, total, paper_id, step="mineru", detail="failed -> skip")
            continue

        prompt = _render_prompt(template, code=code, smiles=smiles)
        context = ""
        user_prompt = ""

        parsed: dict[str, Any] | None = None
        embed_context: str | None = None

        if retrieval == "keyword_first":
            try:
                _log_progress(row_idx, total, paper_id, step="keyword_retrieval", detail="loading chunks")
                all_chunks = _load_chunks(doc_dir, chunk_chars=chunk_chars)
            except Exception as exc:
                result["status"] = "mineru_failed"
                result["error"] = f"Failed to load chunks: {exc}"
                _atomic_write_json(cache_path, result)
                processed += 1
                stats["mineru_failed"] += 1
                _log_progress(row_idx, total, paper_id, step="keyword_retrieval", detail="load failed -> skip")
                continue

            picked = _select_top_chunks(
                all_chunks, code=code, top_k=top_k, max_context_chars=max_context_chars
            )
            context = _render_context(picked)
            _log_progress(
                row_idx,
                total,
                paper_id,
                step="keyword_retrieval",
                detail=f"chunks={len(all_chunks)} picked={len(picked)} context_chars={len(context)}",
            )
            user_prompt = f"{prompt}\n\nContext (extracted from paper):\n{context}\n"

            try:
                _log_progress(row_idx, total, paper_id, step="llm", detail=f"keyword_first model={chat_model}")
                raw = client.chat(
                    model=chat_model,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                result["raw_response"] = raw
                parsed = _parse_llm_json(raw)
                result["method"] = "keyword_first"
            except Exception as exc:
                result["warnings"].append("keyword_first_failed")
                result["error"] = str(exc)

            needs_fallback = parsed is None
            if parsed is not None:
                if verify:
                    susp = _suspicious_output(parsed, context=context)
                    for w in susp:
                        result["warnings"].append(f"keyword_first:{w}")
                    if susp:
                        needs_fallback = True
                if _needs_embedding_fallback(parsed, min_confidence=min_confidence):
                    needs_fallback = True

            if needs_fallback:
                try:
                    _log_progress(row_idx, total, paper_id, step="llm", detail="embedding_fallback (retrieve by embeddings)")
                    t0 = time.time()
                    embed_chunks = _retrieve_by_embedding(
                        doc_dir=doc_dir,
                        provider=provider,
                        embed_model=embed_model,
                        chunk_chars=chunk_chars,
                        top_k=top_k,
                        max_context_chars=max_context_chars,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        code=code,
                        smiles=smiles,
                        rebuild=False,
                    )
                    embed_context = _render_context(embed_chunks)
                    user_prompt2 = f"{prompt}\n\nContext (extracted from paper):\n{embed_context}\n"
                    raw2 = client.chat(
                        model=chat_model,
                        user_prompt=user_prompt2,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                    result["raw_response_fallback"] = raw2
                    parsed = _parse_llm_json(raw2)
                    result["method"] = "embedding_fallback"
                    _log_progress(
                        row_idx,
                        total,
                        paper_id,
                        step="llm",
                        detail=f"embedding_fallback done ({_fmt_duration_s(time.time()-t0)})",
                    )
                except Exception as exc:
                    result["status"] = "llm_failed"
                    if not result["error"]:
                        result["error"] = str(exc)
                    _atomic_write_json(cache_path, result)
                    processed += 1
                    stats["llm_failed"] += 1
                    _log_progress(row_idx, total, paper_id, step="llm", detail="failed -> skip")
                    continue

        elif retrieval == "embedding_only":
            try:
                _log_progress(row_idx, total, paper_id, step="embedding_retrieval", detail="retrieving by embeddings")
                t0 = time.time()
                embed_chunks = _retrieve_by_embedding(
                    doc_dir=doc_dir,
                    provider=provider,
                    embed_model=embed_model,
                    chunk_chars=chunk_chars,
                    top_k=top_k,
                    max_context_chars=max_context_chars,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    code=code,
                    smiles=smiles,
                    rebuild=False,
                )
                embed_context = _render_context(embed_chunks)
                _log_progress(
                    row_idx,
                    total,
                    paper_id,
                    step="embedding_retrieval",
                    detail=f"picked={len(embed_chunks)} context_chars={len(embed_context)} ({_fmt_duration_s(time.time()-t0)})",
                )
            except Exception as exc:
                result["status"] = "llm_failed"
                result["error"] = f"Embedding retrieval failed: {exc}"
                _atomic_write_json(cache_path, result)
                processed += 1
                stats["llm_failed"] += 1
                _log_progress(row_idx, total, paper_id, step="embedding_retrieval", detail="failed -> skip")
                continue

            try:
                _log_progress(row_idx, total, paper_id, step="llm", detail=f"embedding_only model={chat_model}")
                raw = client.chat(
                    model=chat_model,
                    user_prompt=f"{prompt}\n\nContext (extracted from paper):\n{embed_context}\n",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                result["raw_response"] = raw
                parsed = _parse_llm_json(raw)
                result["method"] = "embedding_only"
                if verify:
                    susp = _suspicious_output(parsed, context=embed_context)
                    for w in susp:
                        result["warnings"].append(f"embedding_only:{w}")
            except Exception as exc:
                result["status"] = "llm_failed"
                result["error"] = str(exc)
                _atomic_write_json(cache_path, result)
                processed += 1
                stats["llm_failed"] += 1
                _log_progress(row_idx, total, paper_id, step="llm", detail="failed -> skip")
                continue

        else:
            raise BackfillError(f"Unsupported retrieval: {retrieval}")

        if parsed is None:
            result["status"] = "llm_failed"
            if not result["error"]:
                result["error"] = "No parsed response."
            _atomic_write_json(cache_path, result)
            processed += 1
            stats["llm_failed"] += 1
            _log_progress(row_idx, total, paper_id, step="llm", detail="no parsed output -> skip")
            continue

        result["identity_match"] = parsed.get("identity_match", "")
        result["identity_match_confidence"] = parsed.get("identity_match_confidence", "")

        conf_value = parsed.get("identity_match_confidence")
        if verify and _is_number(conf_value) and float(conf_value) < float(min_confidence):
            result["warnings"].append(f"identity_confidence_below_threshold:{float(conf_value):.3f}")
            solid_nm, solid_warn = None, None
            aggr_nm, aggr_warn = None, None
        else:
            solid_nm, solid_warn = _validate_nm(parsed.get("emission_solid_or_film_nm"), nm_min=nm_min, nm_max=nm_max)
            aggr_nm, aggr_warn = _validate_nm(parsed.get("emission_aggr_nm"), nm_min=nm_min, nm_max=nm_max)
        if solid_warn:
            result["warnings"].append(f"solid:{solid_warn}")
        if aggr_warn:
            result["warnings"].append(f"aggr:{aggr_warn}")

        if verify:
            if embed_context:
                final_context = embed_context
            else:
                final_context = context
            if solid_nm is not None and not _num_appears_in_text(float(solid_nm), final_context):
                result["warnings"].append("solid:not_in_final_context")
                solid_nm = None
            if aggr_nm is not None and not _num_appears_in_text(float(aggr_nm), final_context):
                result["warnings"].append("aggr:not_in_final_context")
                aggr_nm = None

        row["emission_solid"] = _format_nm(solid_nm)
        row["emission_aggr"] = _format_nm(aggr_nm)
        result["emission_solid"] = row["emission_solid"]
        result["emission_aggr"] = row["emission_aggr"]

        result["status"] = "ok"
        _atomic_write_json(cache_path, result)
        processed += 1
        stats["ok"] += 1
        _log_progress(
            row_idx,
            total,
            paper_id,
            step="done",
            detail=f"ok method={result['method']} solid={row['emission_solid'] or 'null'} aggr={row['emission_aggr'] or 'null'}",
        )

    _write_ready_csv_with_emissions(rows=rows, output_csv=csv_path, backup=backup)
    _build_report(cache_dir=cache_dir, report_path=report_path)
    print(f"[run] wrote -> {csv_path}")
    print(f"[run] report -> {report_path}")
    print(
        "[run] summary:"
        f" ok={stats['ok']}"
        f" ok_cached={stats['ok_cached']}"
        f" missing_pdf={stats['missing_pdf']}"
        f" mineru_failed={stats['mineru_failed']}"
        f" llm_failed={stats['llm_failed']}"
    )


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    default_input = here / "no_ready_data" / "rag_compound.csv"
    default_ready = here / "ready_data" / "rag_compound.csv"
    default_pdf_dir = here / "Papers"
    default_prompt = here / "prompts" / "emission_prompt_template.txt"

    p = argparse.ArgumentParser(
        description="Backfill emission_solid/emission_aggr via MinerU outputs + OpenAI (keyword-first, resumable)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Create ready CSV (remove emission columns)")
    p_prepare.add_argument("--input", type=Path, default=default_input)
    p_prepare.add_argument("--output", type=Path, default=default_ready)
    p_prepare.set_defaults(func=lambda a: _prepare_csv(input_csv=a.input, output_csv=a.output))

    p_run = sub.add_parser("run", help="Run backfill and overwrite ready CSV with emissions")
    p_run.add_argument("--csv", type=Path, default=default_ready)
    p_run.add_argument("--pdf-dir", type=Path, default=default_pdf_dir)
    p_run.add_argument("--mineru-output-root", type=Path, default=here.parent / "output")
    p_run.add_argument("--prompt-template", type=Path, default=default_prompt)
    p_run.add_argument("--provider", choices=["openai"], default="openai")
    p_run.add_argument("--retrieval", choices=["keyword_first", "embedding_only"], default="keyword_first")
    p_run.add_argument("--chat-model", default="gpt-5")
    p_run.add_argument("--embed-model", default="text-embedding-3-large")
    p_run.add_argument("--openai-api-key", default=None)
    p_run.add_argument("--openai-base-url", default=None)
    p_run.add_argument("--chunk-chars", type=int, default=1000)
    p_run.add_argument("--top-k", type=int, default=8)
    p_run.add_argument("--max-context-chars", type=int, default=16000)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max-tokens", type=int, default=900)
    p_run.add_argument(
        "--response-format",
        choices=["auto", "json"],
        default="json",
        help="Best-effort JSON output mode for chat models (recommended: json).",
    )
    p_run.add_argument("--min-confidence", type=float, default=0.6)
    p_run.add_argument("--nm-min", type=float, default=200.0)
    p_run.add_argument("--nm-max", type=float, default=1200.0)
    p_run.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Disable value-in-context verification (higher recall, higher hallucination risk).",
    )
    p_run.set_defaults(verify=True)
    p_run.add_argument("--limit", type=int, default=0, help="Process first N rows (0 = no limit)")
    p_run.add_argument("--force", action="store_true", help="Ignore cache and redo LLM calls")
    p_run.add_argument("--mineru-force", action="store_true", help="Force rerun MinerU even if outputs exist")
    p_run.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume-from-cache")
    p_run.set_defaults(resume=True)
    p_run.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create CSV backup")
    p_run.set_defaults(backup=True)
    p_run.set_defaults(func=_cmd_run)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except BackfillError as exc:
        raise SystemExit(f"[error] {exc}") from exc


if __name__ == "__main__":
    main()
