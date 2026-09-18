import os
from src.ingestion.base_loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.common.types import Document

class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        extracted_pages = []
        num_pages = 0

        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())
        except Exception:
            # Simple text fallback if pypdf not installed
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_pages.append(f.read())

        raw_text = "\n\n".join(extracted_pages)
        clean_text = TextCleaner.clean(raw_text)
        doc_id = self._generate_doc_id(file_path)
        doc_name = os.path.basename(file_path)
        approx_tokens = len(clean_text.split())

        return Document(
            document_id=doc_id,
            document_name=doc_name,
            file_path=os.path.abspath(file_path),
            file_type="pdf",
            raw_text=raw_text,
            clean_text=clean_text,
            total_tokens=approx_tokens,
            metadata={"num_pages": num_pages, "character_count": len(clean_text)}
        )
