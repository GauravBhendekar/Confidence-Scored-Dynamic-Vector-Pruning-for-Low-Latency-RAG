import os
from src.ingestion.base_loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.common.types import Document

class TXTLoader(DocumentLoader):
    def load(self, file_path: str) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        clean_text = TextCleaner.clean(raw_text)
        doc_id = self._generate_doc_id(file_path)
        doc_name = os.path.basename(file_path)
        approx_tokens = len(clean_text.split())

        return Document(
            document_id=doc_id,
            document_name=doc_name,
            file_path=os.path.abspath(file_path),
            file_type="txt",
            raw_text=raw_text,
            clean_text=clean_text,
            total_tokens=approx_tokens,
            metadata={"character_count": len(clean_text)}
        )
