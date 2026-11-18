"""Structure-aware ingestion plus persistent lexical and vector indexes."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from .config import KNOWLEDGE_DIR
from .llm import embed_documents

POLICY_URL = "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2"
CHUNK_WORDS = 360
CHUNK_OVERLAP = 60


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: str
    text: str
    embedding: list[float] | None = None
    page: int | None = None
    criterion_id: str | None = None


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _criterion(section: str) -> str | None:
    match = re.search(r"criterion\s*(10|[1-9])\b", section, re.I)
    return f"C{match.group(1)}" if match else None


def _heading(line: str) -> bool:
    line = line.strip()
    return bool(re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:section\s+[a-z]|criterion\s+\d+|step\s+[12]|comparable evidence|final merits determination|other evidentiary considerations|overview|eligibility)\b", line, re.I))


def _split_units(text: str) -> list[tuple[str, str]]:
    """Return (section heading, paragraph) units while retaining heading context."""
    active = "USCIS Policy Manual, Volume 6, Part F, Chapter 2"
    units = []
    for raw in re.split(r"\n+", text):
        line = clean_text(raw)
        if not line:
            continue
        if _heading(line):
            active = line[:180]
        # Extracted PDF tables often arrive as short lines; keep them with the active section.
        units.append((active, line))
    if not units:
        return [(active, clean_text(text))]
    return units


def split_hybrid_chunks(text: str, source: str, page: int | None = None, prefix: str = "chunk") -> list[Chunk]:
    """Split at headings/paragraphs, then pack to a target size with word overlap.

    Unlike fixed-window chunking, units from different detected sections are never
    combined. Oversized paragraphs are divided with overlap.
    """
    units = _split_units(text)
    output: list[Chunk] = []
    current_section = None
    current_words: list[str] = []

    def flush():
        nonlocal current_words
        if not current_words:
            return
        text_part = " ".join(current_words)
        idx = len(output)
        section = current_section or "USCIS Policy Manual, Volume 6, Part F, Chapter 2"
        output.append(Chunk(f"{prefix}-{idx:04d}", source, section, text_part, page=page, criterion_id=_criterion(section)))
        current_words = []

    for section, paragraph in units:
        words = paragraph.split()
        if current_section is not None and section != current_section:
            flush()
        current_section = section
        if len(words) > CHUNK_WORDS:
            flush()
            stride = CHUNK_WORDS - CHUNK_OVERLAP
            for start in range(0, len(words), stride):
                part = words[start:start + CHUNK_WORDS]
                if not part:
                    break
                section_name = section
                output.append(Chunk(f"{prefix}-{len(output):04d}", source, section_name, " ".join(part), page=page, criterion_id=_criterion(section_name)))
                if start + CHUNK_WORDS >= len(words):
                    break
            continue
        if len(current_words) + len(words) > CHUNK_WORDS:
            previous = current_words[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            flush()
            current_words = previous
        current_words.extend(words)
    flush()
    return output


def split_chunks(text: str, size: int = 1100, overlap: int = 140) -> list[str]:
    """Backward-compatible word-window helper."""
    words = clean_text(text).split()
    result, start = [], 0
    while start < len(words):
        end = min(len(words), start + size)
        result.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return result


def ingest_url(url: str = POLICY_URL) -> list[Chunk]:
    response = requests.get(url, timeout=30, headers={"User-Agent": "EB1A-Profile-Evaluator/0.2 educational prototype"})
    response.raise_for_status()
    if url.lower().endswith(".pdf") or "application/pdf" in response.headers.get("Content-Type", ""):
        import io
        pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages]
        return [chunk for number, page in enumerate(pages, start=1) for chunk in split_hybrid_chunks(page, url, number, f"uscis-p{number}")]
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    chunks = split_hybrid_chunks(soup.get_text("\n"), url, prefix="uscis")
    if sum(len(c.text) for c in chunks) < 300:
        raise ValueError("The page returned too little readable text. USCIS may have blocked retrieval.")
    return chunks


def chunks_from_local_files(folder: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(folder.glob("**/*")):
        if not path.is_file() or path.name in {"chunks.json", "manifest.json"} or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                for number, page in enumerate(PdfReader(str(path)).pages, start=1):
                    text = page.extract_text() or ""
                    chunks.extend(split_hybrid_chunks(text, path.name, number, f"{path.stem}-p{number}"))
            else:
                text = path.read_text(encoding="utf-8")
                chunks.extend(split_hybrid_chunks(text, path.name, prefix=path.stem))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    return chunks


def save_chunks(chunks: Iterable[Chunk], path: Path | None = None) -> Path:
    path = path or KNOWLEDGE_DIR / "chunks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or KNOWLEDGE_DIR / "chunks.json"
    if not path.exists():
        raise FileNotFoundError("Knowledge base is empty. Run `python -m eb1a_evaluator.cli ingest --local` first.")
    return [Chunk(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _embed(texts: list[str]) -> list[list[float]] | None:
    return embed_documents(texts)


def embed_and_save(chunks: list[Chunk]) -> Path:
    vectors = _embed([c.text for c in chunks])
    if vectors:
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
    path = save_chunks(chunks)
    from .retrieval import build_chroma_index, build_vector_index, BM25Index
    BM25Index(chunks).save(KNOWLEDGE_DIR / "bm25.index.json")
    build_chroma_index(chunks, KNOWLEDGE_DIR / "chroma_db")
    build_vector_index(chunks, KNOWLEDGE_DIR / "hnsw.index")
    return path
