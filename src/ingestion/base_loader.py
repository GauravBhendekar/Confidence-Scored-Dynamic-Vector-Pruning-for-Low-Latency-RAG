from abc import ABC, abstractmethod
import os
import uuid
from src.common.types import Document

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> Document:
        pass

    def _generate_doc_id(self, file_path: str) -> str:
        basename = os.path.basename(file_path)
        return f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, basename).hex[:8]}"
