
import pytest

from ingestion.chunking.base import DocumentChunk
from ingestion.chunking.config import ChunkingConfig
from ingestion.chunking.text_chunker import TextChunker
from ingestion.loaders.base import DocumentPage, LoadedDocument


class FakeTokenizer:
    """Simple whitespace tokenizer for deterministic unit tests."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(token_ids)


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def chunker(tokenizer):
    return TextChunker(
        config=ChunkingConfig(
            chunk_size=10,
            chunk_overlap=2,
        ),
        tokenizer=tokenizer,
    )


def make_document(pages, source_format="txt"):
    return LoadedDocument(
        source_format=source_format,
        page_count=len(pages) if source_format == "pdf" else None,
        pages=[
            DocumentPage(page_number=page_number, text=text)
            for page_number, text in pages
        ],
    )


# ==========================================
# Basic chunking
# ==========================================

def test_chunker_returns_document_chunks(chunker):
    document = make_document([
        (None, "one two three four five six seven eight nine ten"),
    ])

    result = chunker.chunk_document(document)

    assert len(result) == 1
    assert isinstance(result[0], DocumentChunk)


def test_chunker_respects_maximum_token_count(chunker):
    text = " ".join(f"word{i}" for i in range(25))
    document = make_document([(None, text)])

    result = chunker.chunk_document(document)

    assert len(result) > 1

    for chunk in result:
        assert chunk.token_count <= 10


def test_chunker_assigns_sequential_indexes(chunker):
    text = " ".join(f"word{i}" for i in range(25))
    document = make_document([(None, text)])

    result = chunker.chunk_document(document)

    assert [chunk.chunk_index for chunk in result] == list(
        range(len(result))
    )



def test_chunker_preserves_all_content(chunker):
    words = [f"word{i}" for i in range(25)]
    document = make_document([(None, " ".join(words))])

    result = chunker.chunk_document(document)

    reconstructed = []

    for index, chunk in enumerate(result):
        chunk_words = chunk.content.split()

        if index > 0:
            overlap = chunker.config.chunk_overlap
            chunk_words = chunk_words[overlap:]

        reconstructed.extend(chunk_words)

    assert reconstructed == words


# ==========================================
# Overlap
# ==========================================

def test_chunker_preserves_overlap_between_chunks(chunker):
    words = [f"word{i}" for i in range(20)]
    document = make_document([(None, " ".join(words))])

    result = chunker.chunk_document(document)

    assert len(result) > 1

    for previous, current in zip(result, result[1:]):
        previous_words = previous.content.split()
        current_words = current.content.split()

        assert previous_words[-2:] == current_words[:2]


# ==========================================
# Page references
# ==========================================

def test_chunker_preserves_pdf_page_numbers(chunker):
    document = make_document(
        [
            (1, " ".join(f"page1word{i}" for i in range(15))),
            (2, " ".join(f"page2word{i}" for i in range(15))),
        ],
        source_format="pdf",
    )

    result = chunker.chunk_document(document)

    page_one_chunks = [
        chunk for chunk in result if chunk.page_start == 1
    ]
    page_two_chunks = [
        chunk for chunk in result if chunk.page_start == 2
    ]

    assert page_one_chunks
    assert page_two_chunks

    assert all(
        chunk.page_start == chunk.page_end == 1
        for chunk in page_one_chunks
    )

    assert all(
        chunk.page_start == chunk.page_end == 2
        for chunk in page_two_chunks
    )


def test_chunker_preserves_none_page_numbers(chunker):
    document = make_document([
        (None, "one two three four five"),
    ])

    result = chunker.chunk_document(document)

    assert result[0].page_start is None
    assert result[0].page_end is None


# ==========================================
# Empty and short documents
# ==========================================

def test_chunker_handles_empty_document(chunker):
    document = make_document([])

    assert chunker.chunk_document(document) == []


def test_chunker_handles_empty_pages(chunker):
    document = make_document([
        (None, ""),
        (None, "   "),
    ])

    assert chunker.chunk_document(document) == []


def test_chunker_handles_document_smaller_than_chunk_size(chunker):
    document = make_document([
        (None, "one two three four five"),
    ])

    result = chunker.chunk_document(document)

    assert len(result) == 1
    assert result[0].token_count == 5


def test_chunker_handles_exact_chunk_size(chunker):
    document = make_document([
        (None, "one two three four five six seven eight nine ten"),
    ])

    result = chunker.chunk_document(document)

    assert len(result) == 1
    assert result[0].token_count == 10


# ==========================================
# Multilingual content
# ==========================================

@pytest.mark.parametrize(
    "text",
    [
        "Machine learning and artificial intelligence",
        "Apprentissage automatique et intelligence artificielle",
        "الذكاء الاصطناعي وتحليل البيانات",
    ],
)
def test_chunker_accepts_multilingual_text(chunker, text):
    document = make_document([(None, text)])

    result = chunker.chunk_document(document)

    assert len(result) == 1
    assert result[0].content


# ==========================================
# Configuration validation
# ==========================================

def test_chunker_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)


def test_chunker_rejects_negative_overlap():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_overlap=-1)


def test_chunker_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_chunker_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=11)
