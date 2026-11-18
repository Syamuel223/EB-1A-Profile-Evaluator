import argparse
from pathlib import Path
from pypdf import PdfReader
from .config import KNOWLEDGE_DIR, SOURCE_URL
from .knowledge import ingest_url, chunks_from_local_files, embed_and_save, load_chunks
from .pipeline import assess_profile
from .opportunities import suggest_opportunities
import json


def read_profile(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(prog="eb1a", description="Cited EB-1A profile screening prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Fetch and index the USCIS chapter")
    ing.add_argument("--url", default=SOURCE_URL)
    ing.add_argument("--local", action="store_true", help="Index files already placed in data/knowledge_base")
    ev = sub.add_parser("evaluate", help="Evaluate a PDF or text profile and print JSON")
    ev.add_argument("profile", type=Path)
    sub.add_parser("sources", help="List indexed source chunks")
    opp = sub.add_parser("opportunities", help="Research live opportunities for weak criteria from a saved assessment JSON")
    opp.add_argument("assessment", type=Path)
    opp.add_argument("--field", required=True)
    args = parser.parse_args()
    if args.command == "ingest":
        chunks = chunks_from_local_files(KNOWLEDGE_DIR) if args.local else ingest_url(args.url)
        path = embed_and_save(chunks)
        print(f"Indexed {len(chunks)} chunks at {path}")
    elif args.command == "evaluate":
        print(assess_profile(read_profile(args.profile)).model_dump_json(indent=2))
    elif args.command == "sources":
        for chunk in load_chunks():
            print(f"{chunk.chunk_id}\t{chunk.source}\t{chunk.section}\t{chunk.text[:180]}")
    elif args.command == "opportunities":
        assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
        print(json.dumps(suggest_opportunities(assessment, args.field), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
