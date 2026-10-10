from src.chunking.models import ChunkingConfig
from src.chunking.recursive_chunker import RecursivePageAwareChunker
from src.chunking.validate_chunks import validate


def make_document():
    paragraph = "Nordic climate policy supports emissions reductions through coordinated measures. " * 22
    return {
        "document_id": "doc_999",
        "metadata": {
            "title": "Synthetic report",
            "year": 2025,
            "organization": "Nordic Council of Ministers",
            "country": "Nordic region",
            "topic": "climate-policy",
            "secondary_topics": "energy; decarbonisation",
            "source_url": "https://example.org/report",
        },
        "parser": {"name": "pymupdf", "page_count": 2},
        "pages": [
            {"page_number": 1, "text": paragraph + "\n\n" + paragraph, "char_count": 100},
            {"page_number": 2, "text": paragraph + "\n\n" + paragraph, "char_count": 100},
        ],
    }


def test_chunk_ids_metadata_and_pages_are_traceable():
    chunker = RecursivePageAwareChunker(ChunkingConfig(chunk_size=120, chunk_overlap=20, minimum_chunk_size=30))
    chunks = chunker.chunk_document(make_document())

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "doc_999_chunk_0001"
    assert chunks[0].document_id == "doc_999"
    assert chunks[0].topic == "climate-policy"
    assert chunks[0].secondary_topics == ["energy", "decarbonisation"]
    assert all(c.page_start <= c.page_end for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_validator_accepts_generated_chunks():
    chunker = RecursivePageAwareChunker(ChunkingConfig(chunk_size=120, chunk_overlap=20, minimum_chunk_size=30))
    rows = [c.to_dict() for c in chunker.chunk_document(make_document())]
    result = validate(rows)
    assert result["ok"] is True
    assert result["document_count"] == 1
