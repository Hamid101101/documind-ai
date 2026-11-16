
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DocumentPage:
    page_number: int | None
    text: str


@dataclass
class LoadedDocument:
    source_format: str
    page_count: int | None
    pages: list[DocumentPage]


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> LoadedDocument:
        """Extract text from a document."""
        raise NotImplementedError
