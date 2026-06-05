from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.base_models import InputFormat
from pathlib import Path


class DoclingProcessor:
    """
    Docling برای:
    - استخراج ساختار از PDF / DOCX / HTML
    - جداول + تصاویر + فرمول
    - chunking هوشمند با حفظ context
    """

    def __init__(self, config: dict):
        options = PipelineOptions(
            do_ocr=config.get("docling_ocr", True),
            do_table_structure=True,
        )
        self.converter = DocumentConverter(pipeline_options=options)
        self.chunk_size    = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)

    def process_file(self, file_path: str) -> list[dict]:
        """تبدیل فایل به chunk های ساختاریافته"""
        result = self.converter.convert(file_path)
        doc = result.document

        chunks = []

        # متن اصلی
        for i, chunk in enumerate(
            doc.chunked_texts(
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap,
            )
        ):
            chunks.append({
                "id":       f"{Path(file_path).stem}_text_{i}",
                "content":  chunk.text,
                "type":     "text",
                "page":     chunk.meta.get("page"),
                "source":   file_path,
                "metadata": {"section": chunk.meta.get("section", "")},
            })

        # جداول (به صورت مجزا - ارزش اطلاعاتی بالا)
        for i, table in enumerate(doc.tables):
            chunks.append({
                "id":      f"{Path(file_path).stem}_table_{i}",
                "content": table.export_to_markdown(),
                "type":    "table",
                "source":  file_path,
                "metadata": {"caption": table.caption or ""},
            })

        return chunks

    def process_url(self, url: str) -> list[dict]:
        """پردازش مستقیم URL"""
        result = self.converter.convert(url)
        return self.process_file.__wrapped__(self, result)
