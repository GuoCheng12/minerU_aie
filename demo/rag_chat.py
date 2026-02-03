import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page_start: int | None
    page_end: int | None
    text: str


class RAGError(RuntimeError):
    pass


DEFAULT_SYSTEM_PROMPT = (
    "你是一个文献问答助手。请严格基于提供的上下文回答；如果上下文没有明确提到，就回答“不确定/未在文中找到”。"
    "默认用用户提问的语言作答。回答时尽量给出引用页码（例如 p3 或 p3-4）。"
)


def _load_existing_index_meta(index_dir: Path) -> dict[str, Any] | None:
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _apply_runtime_defaults(args: argparse.Namespace, *, index_dir: Path | None) -> None:
    meta = _load_existing_index_meta(index_dir) if index_dir else None

    if getattr(args, "provider", None) is None and meta and isinstance(meta.get("provider"), str):
        args.provider = meta["provider"]

    if getattr(args, "embed_model", None) is None and meta and isinstance(meta.get("embed_model"), str):
        args.embed_model = meta["embed_model"]

    if getattr(args, "system_prompt", None) is None and meta and isinstance(meta.get("system_prompt"), str):
        args.system_prompt = meta["system_prompt"]

    if getattr(args, "provider", None) is None:
        args.provider = "ollama"

    if getattr(args, "embed_model", None) is None:
        args.embed_model = "nomic-embed-text" if args.provider == "ollama" else "text-embedding-3-small"

    if getattr(args, "chat_model", None) is None:
        args.chat_model = "qwen2.5:7b-instruct" if args.provider == "ollama" else "gpt-4o-mini"


def _resolve_path_relative_to_doc(doc: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = doc if doc.is_dir() else doc.parent
    return (base / p).resolve()


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RAGError(f"File not found: {path}") from exc
    except Exception as exc:
        raise RAGError(f"Failed to read file: {path}") from exc


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
        return html

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


def _load_chunks_from_content_list_v2(path: Path, *, chunk_chars: int) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RAGError(f"Invalid content_list_v2 format: expected list, got {type(data)}")

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
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RAGError(f"Invalid content_list format: expected list, got {type(data)}")

    units: list[tuple[int | None, str]] = []
    for block in data:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"page_header", "page_footer", "page_number"}:
            continue
        page_idx = block.get("page_idx")
        page_no = (page_idx + 1) if isinstance(page_idx, int) else None
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            units.append((page_no, text.strip()))
            continue
        if block_type == "image":
            caption = "".join(
                _render_inline(x) for x in _iter_content_items(block.get("image_caption"))
            ).strip()
            if caption:
                units.append((page_no, f"Figure: {caption}"))

    return _merge_units_to_chunks(units, chunk_chars=chunk_chars)


def _load_chunks_from_md(path: Path, *, chunk_chars: int) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
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


def _merge_units_to_chunks(units: list[tuple[int | None, str]], *, chunk_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    def split_long_text(text: str) -> list[str]:
        text = text.strip()
        if len(text) <= chunk_chars:
            return [text]
        # Chunk by characters with a small overlap, preferring whitespace boundaries.
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


def _resolve_source_path(doc: Path) -> tuple[Path, str]:
    if doc.is_dir():
        v2 = sorted(doc.glob("*_content_list_v2.json"))
        if v2:
            return v2[0], "content_list_v2"
        v1 = sorted(doc.glob("*_content_list.json"))
        if v1:
            return v1[0], "content_list"
        mds = sorted(doc.glob("*.md"))
        if mds:
            return mds[0], "md"
        raise RAGError(f"No supported source file found in: {doc}")

    if doc.name.endswith("_content_list_v2.json"):
        return doc, "content_list_v2"
    if doc.name.endswith("_content_list.json"):
        return doc, "content_list"
    if doc.suffix.lower() == ".md":
        return doc, "md"
    raise RAGError(f"Unsupported source file: {doc}")


def _default_index_dir(source_path: Path) -> Path:
    if source_path.is_dir():
        return source_path / ".rag_index"
    return source_path.parent / ".rag_index"


def _load_chunks(source_path: Path, source_kind: str, *, chunk_chars: int) -> list[Chunk]:
    if source_kind == "content_list_v2":
        return _load_chunks_from_content_list_v2(source_path, chunk_chars=chunk_chars)
    if source_kind == "content_list":
        return _load_chunks_from_content_list_v1(source_path, chunk_chars=chunk_chars)
    if source_kind == "md":
        return _load_chunks_from_md(source_path, chunk_chars=chunk_chars)
    raise RAGError(f"Unknown source kind: {source_kind}")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec)) + 1e-12
    return vec / denom


class OllamaClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def embed(self, *, model: str, text: str, timeout_s: float = 120.0) -> np.ndarray:
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:
            raise RAGError("Missing dependency: httpx. Please run in MinerU virtualenv.") from exc

        payload = {"model": model, "prompt": text}
        url = f"{self._base_url}/api/embeddings"
        try:
            resp = httpx.post(url, json=payload, timeout=timeout_s)
        except Exception:
            alt = f"{self._base_url}/api/embed"
            resp = httpx.post(alt, json={"model": model, "input": text}, timeout=timeout_s)

        if resp.status_code != 200:
            raise RAGError(f"Ollama embeddings failed: HTTP {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        emb = data.get("embedding")
        if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
            return np.asarray(emb, dtype=np.float32)
        # /api/embed may return {"embeddings":[{"embedding":[...]}]}
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs:
            first = embs[0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return np.asarray(first["embedding"], dtype=np.float32)
        raise RAGError("Ollama embeddings response missing embedding vector.")

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        timeout_s: float = 300.0,
    ) -> str:
        try:
            import httpx  # type: ignore[import-not-found]
        except Exception as exc:
            raise RAGError("Missing dependency: httpx. Please run in MinerU virtualenv.") from exc

        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        resp = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            raise RAGError(f"Ollama chat failed: HTTP {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        msg = data.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"].strip()
        raise RAGError("Ollama chat response missing assistant message.")


class OpenAIClient:
    def __init__(self, *, api_key: str, base_url: str | None = None):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:
            raise RAGError("Missing dependency: openai. Please run in MinerU virtualenv.") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def embed(self, *, model: str, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=model, input=text)
        emb = resp.data[0].embedding
        return np.asarray(emb, dtype=np.float32)

    def chat(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        msg = resp.choices[0].message
        return (msg.content or "").strip()


def _cosine_top_k(embeddings: np.ndarray, query: np.ndarray, *, top_k: int) -> list[int]:
    if embeddings.ndim != 2:
        raise RAGError(f"Invalid embeddings matrix shape: {embeddings.shape}")
    if query.ndim != 1:
        raise RAGError(f"Invalid query vector shape: {query.shape}")
    if embeddings.shape[1] != query.shape[0]:
        raise RAGError(
            f"Embedding dim mismatch: index dim {embeddings.shape[1]} vs query dim {query.shape[0]}"
        )
    sims = embeddings @ query
    top_k = min(max(int(top_k), 1), embeddings.shape[0])
    idx = np.argpartition(-sims, kth=top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return [int(i) for i in idx.tolist()]


def _index_paths(index_dir: Path) -> tuple[Path, Path, Path]:
    return index_dir / "meta.json", index_dir / "chunks.jsonl", index_dir / "embeddings.npy"


def _build_index(
    *,
    source_path: Path,
    source_kind: str,
    index_dir: Path,
    provider: str,
    embed_model: str,
    chunk_chars: int,
    rebuild: bool,
    dry_run: bool,
    ollama_url: str,
    openai_api_key: str | None,
    openai_base_url: str | None,
) -> tuple[list[Chunk], np.ndarray | None]:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path, chunks_path, embeddings_path = _index_paths(index_dir)

    stat = source_path.stat()
    source_mtime = int(stat.st_mtime)
    source_size = int(stat.st_size)

    if not rebuild and meta_path.exists() and chunks_path.exists() and embeddings_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("source_path") == str(source_path)
                and int(meta.get("source_mtime", -1)) == source_mtime
                and int(meta.get("source_size", -1)) == source_size
                and meta.get("provider") == provider
                and meta.get("embed_model") == embed_model
                and int(meta.get("chunk_chars", -1)) == int(chunk_chars)
            ):
                chunks = _read_chunks_jsonl(chunks_path)
                embeddings = np.load(embeddings_path)
                return chunks, embeddings
        except Exception:
            pass

    chunks = _load_chunks(source_path, source_kind, chunk_chars=chunk_chars)
    if dry_run:
        return chunks, None

    if provider == "ollama":
        client: Any = OllamaClient(ollama_url)
    elif provider == "openai":
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RAGError("OPENAI_API_KEY is required for provider=openai")
        base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        client = OpenAIClient(api_key=api_key, base_url=base_url)
    else:
        raise RAGError(f"Unsupported provider: {provider}")

    embeddings: list[np.ndarray] = []
    for i, ch in enumerate(chunks, start=1):
        print(f"[index] embedding {i}/{len(chunks)}", file=sys.stderr)
        vec = client.embed(model=embed_model, text=ch.text)
        embeddings.append(_l2_normalize(vec))

    mat = np.vstack(embeddings).astype(np.float32, copy=False)
    _write_chunks_jsonl(chunks_path, chunks)
    np.save(embeddings_path, mat)

    meta = {
        "version": 1,
        "source_path": str(source_path),
        "source_kind": source_kind,
        "source_mtime": source_mtime,
        "source_size": source_size,
        "provider": provider,
        "embed_model": embed_model,
        "chunk_chars": int(chunk_chars),
        "num_chunks": int(len(chunks)),
        "embedding_dim": int(mat.shape[1]),
        "built_at": int(time.time()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks, mat


def _write_chunks_jsonl(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": ch.chunk_id,
                        "page_start": ch.page_start,
                        "page_end": ch.page_end,
                        "text": ch.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=int(obj["chunk_id"]),
                    page_start=obj.get("page_start"),
                    page_end=obj.get("page_end"),
                    text=str(obj["text"]),
                )
            )
    return chunks


def _format_pages(ch: Chunk) -> str:
    if ch.page_start is None or ch.page_end is None:
        return "p?"
    if ch.page_start == ch.page_end:
        return f"p{ch.page_start}"
    return f"p{ch.page_start}-{ch.page_end}"


def _build_prompt(question: str, retrieved: list[Chunk], *, system_prompt: str | None = None) -> list[dict[str, str]]:
    context_parts: list[str] = []
    for i, ch in enumerate(retrieved, start=1):
        context_parts.append(f"[Context {i} | {_format_pages(ch)}]\n{ch.text}")

    context_text = "\n\n".join(context_parts).strip()
    system = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    user = f"问题：{question}\n\n上下文：\n{context_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_ask(
    *,
    question: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    provider: str,
    chat_model: str,
    embed_model: str,
    top_k: int,
    temperature: float,
    system_prompt: str | None,
    ollama_url: str,
    openai_api_key: str | None,
    openai_base_url: str | None,
) -> tuple[str, list[Chunk]]:
    if provider == "ollama":
        client: Any = OllamaClient(ollama_url)
    elif provider == "openai":
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RAGError("OPENAI_API_KEY is required for provider=openai")
        base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        client = OpenAIClient(api_key=api_key, base_url=base_url)
    else:
        raise RAGError(f"Unsupported provider: {provider}")

    q_vec = _l2_normalize(client.embed(model=embed_model, text=question))
    idxs = _cosine_top_k(embeddings, q_vec, top_k=top_k)
    retrieved = [chunks[i] for i in idxs]
    messages = _build_prompt(question, retrieved, system_prompt=system_prompt)
    answer = client.chat(model=chat_model, messages=messages, temperature=temperature)
    return answer, retrieved


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.doc:
        raise RAGError("--doc is required")
    doc = Path(args.doc).expanduser().resolve()
    source_path, source_kind = _resolve_source_path(doc)
    index_dir = Path(args.index_dir).expanduser().resolve() if args.index_dir else _default_index_dir(source_path)
    _apply_runtime_defaults(args, index_dir=index_dir)

    chunks, _ = _build_index(
        source_path=source_path,
        source_kind=source_kind,
        index_dir=index_dir,
        provider=args.provider,
        embed_model=args.embed_model,
        chunk_chars=args.chunk_chars,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
        ollama_url=args.ollama_url,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )

    print(f"source: {source_path}")
    print(f"index_dir: {index_dir}")
    print(f"chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0].text[:120].replace("\n", " ")
        print(f"sample: {_format_pages(chunks[0])} {sample}...")
    if args.dry_run:
        print("dry_run: skipped embeddings")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if not args.doc:
        raise RAGError("--doc is required")
    doc = Path(args.doc).expanduser().resolve()
    source_path, source_kind = _resolve_source_path(doc)
    index_dir = Path(args.index_dir).expanduser().resolve() if args.index_dir else _default_index_dir(source_path)
    _apply_runtime_defaults(args, index_dir=index_dir)

    if not getattr(args, "system_prompt", None) and getattr(args, "system_prompt_file", None):
        sp_path = _resolve_path_relative_to_doc(doc, args.system_prompt_file)
        args.system_prompt = _read_text_file(sp_path)

    if not getattr(args, "question", None) and getattr(args, "question_file", None):
        q_path = _resolve_path_relative_to_doc(doc, args.question_file)
        args.question = _read_text_file(q_path)

    if not getattr(args, "question", None):
        raise RAGError("Either -q/--question or --question-file/--prompt-file is required.")

    chunks, embeddings = _build_index(
        source_path=source_path,
        source_kind=source_kind,
        index_dir=index_dir,
        provider=args.provider,
        embed_model=args.embed_model,
        chunk_chars=args.chunk_chars,
        rebuild=args.rebuild,
        dry_run=False,
        ollama_url=args.ollama_url,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    if embeddings is None:
        raise RAGError("Index build returned no embeddings.")

    answer, retrieved = _run_ask(
        question=args.question,
        chunks=chunks,
        embeddings=embeddings,
        provider=args.provider,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        temperature=args.temperature,
        system_prompt=getattr(args, "system_prompt", None),
        ollama_url=args.ollama_url,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    print(answer)
    print("\nSources:", ", ".join(_format_pages(ch) for ch in retrieved))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    if not args.doc:
        raise RAGError("--doc is required")
    doc = Path(args.doc).expanduser().resolve()
    source_path, source_kind = _resolve_source_path(doc)
    index_dir = Path(args.index_dir).expanduser().resolve() if args.index_dir else _default_index_dir(source_path)
    _apply_runtime_defaults(args, index_dir=index_dir)

    if not getattr(args, "system_prompt", None) and getattr(args, "system_prompt_file", None):
        sp_path = _resolve_path_relative_to_doc(doc, args.system_prompt_file)
        args.system_prompt = _read_text_file(sp_path)

    chunks, embeddings = _build_index(
        source_path=source_path,
        source_kind=source_kind,
        index_dir=index_dir,
        provider=args.provider,
        embed_model=args.embed_model,
        chunk_chars=args.chunk_chars,
        rebuild=args.rebuild,
        dry_run=False,
        ollama_url=args.ollama_url,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    if embeddings is None:
        raise RAGError("Index build returned no embeddings.")

    print("Enter questions (Ctrl-D/Ctrl-C to exit).")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not q:
            continue
        answer, retrieved = _run_ask(
            question=q,
            chunks=chunks,
            embeddings=embeddings,
            provider=args.provider,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            top_k=args.top_k,
            temperature=args.temperature,
            system_prompt=getattr(args, "system_prompt", None),
            ollama_url=args.ollama_url,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
        )
        print("\nA>", answer)
        print("Sources:", ", ".join(_format_pages(ch) for ch in retrieved))


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--doc", default=None, help="MinerU output dir or a *_content_list(_v2).json/*.md file")
    common.add_argument("--index-dir", default=None, help="Index dir (default: <doc>/.rag_index)")
    common.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM+Embeddings provider (default: reuse existing index; otherwise ollama).",
    )
    common.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base url")
    common.add_argument("--openai-api-key", default=None, help="OpenAI API key (or env OPENAI_API_KEY)")
    common.add_argument("--openai-base-url", default=None, help="OpenAI base url (or env OPENAI_BASE_URL)")
    common.add_argument(
        "--embed-model",
        default=None,
        help="Embeddings model name (default: reuse existing index; otherwise nomic-embed-text for ollama or text-embedding-3-small for openai).",
    )
    common.add_argument(
        "--chat-model",
        default=None,
        help='Chat model name (default: qwen2.5:7b-instruct for ollama or gpt-4o-mini for openai).',
    )
    common.add_argument("--chunk-chars", type=int, default=1800, help="Chunk size by characters")
    common.add_argument("--top-k", type=int, default=6, help="Number of retrieved chunks")
    common.add_argument("--temperature", type=float, default=0.2, help="Chat temperature")
    common.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    common.add_argument("--system-prompt", default=None, help="Override system prompt")
    common.add_argument(
        "--system-prompt-file",
        default=None,
        help="Load system prompt from a file (relative to --doc if not absolute)",
    )

    p = argparse.ArgumentParser(
        description="Simple local RAG chat for MinerU outputs (content_list_v2/content_list/md).",
        parents=[common],
    )

    sub = p.add_subparsers(dest="cmd", required=False)
    p_build = sub.add_parser("build", parents=[common], help="Build (or reuse) index")
    p_build.add_argument("--dry-run", action="store_true", help="Only parse and chunk; skip embeddings")
    p_build.set_defaults(func=_cmd_build)

    p_ask = sub.add_parser("ask", parents=[common], help="Ask a single question")
    p_ask.add_argument("-q", "--question", default=None, help="Question (prompt)")
    p_ask.add_argument(
        "--question-file",
        dest="question_file",
        default=None,
        help="Load question (prompt) from a file (relative to --doc if not absolute)",
    )
    p_ask.add_argument(
        "--prompt-file",
        dest="question_file",
        default=None,
        help="Alias of --question-file",
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", parents=[common], help="Interactive chat")
    p_chat.set_defaults(func=_cmd_chat)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        args.cmd = "chat"
        args.func = _cmd_chat
    try:
        return int(args.func(args))
    except RAGError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
