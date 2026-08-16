from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from src.parsing import ComplexPDFParser

logger = CustomLogger().get_logger(__name__)



class MultimodalDocumentIngestion:
    """Parse, prepare, embed and upsert MM-RAG documents into Qdrant."""

    TEXT_CONTENT_TYPES = {"page_text_plus_ocr", "text"}

    def __init__(
        self,
        collection_name: str | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 120,
        batch_size: int = 64,
    ) -> None:
        try:
            self.collection_name = (
                collection_name
                or os.getenv("QDRANT_COLLECTION_NAME")
                or "mm-rag-documents"
            )
            self.embedding_model = (
                os.getenv("OPENAI_EMBEDDING_MODEL")
                or "text-embedding-3-large"
            )
            self.embedding_dimension = int(
                os.getenv("OPENAI_EMBEDDING_DIMENSION") or 3072
            )
            self.qdrant_url = (
                os.getenv("QDRANT_URL")
                or os.getenv("QDRANT_Cluster_Endpoint")
            )
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
            self.batch_size = batch_size

            self._validate_config()

            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                dimensions=self.embedding_dimension,
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )

            self._ensure_collection()

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                retrieval_mode=RetrievalMode.DENSE,
            )

            logger.info(
                "ingestion_initialized",
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
            )

        except Exception as exc:
            logger.exception("ingestion_initialization_failed", error=str(exc))
            raise DocumentPortalException(
                "Failed to initialize ingestion pipeline", exc
            ) from exc

    def _validate_config(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is missing")
        if self.embedding_dimension <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSION must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _ensure_collection(self) -> None:
        """Create the collection and filter indexes before ingestion."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "qdrant_collection_created",
                collection_name=self.collection_name,
            )

        info = self.client.get_collection(self.collection_name)
        existing_indexes = set((info.payload_schema or {}).keys())

        required_indexes = {
            "metadata.document_id": models.PayloadSchemaType.KEYWORD,
            "metadata.filename": models.PayloadSchemaType.KEYWORD,
            "metadata.file_sha256": models.PayloadSchemaType.KEYWORD,
            "metadata.content_type": models.PayloadSchemaType.KEYWORD,
            "metadata.page_number": models.PayloadSchemaType.INTEGER,
        }

        for field_name, field_schema in required_indexes.items():
            if field_name not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )

    @staticmethod
    def _document_id(source: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"mm-rag-document:{source}"))

    @staticmethod
    def _point_id(
        source: str,
        page_number: Any,
        content_type: str,
        chunk_index: int,
        table_index: Any = "",
        image_index: Any = "",
    ) -> str:
        identity = "|".join(
            [
                source,
                str(page_number),
                content_type,
                str(table_index),
                str(image_index),
                str(chunk_index),
            ]
        )
        return str(uuid5(NAMESPACE_URL, f"mm-rag-point:{identity}"))

    def _split(self, document: Document) -> list[Document]:
        content_type = str(document.metadata.get("content_type", "unknown"))
        if content_type in self.TEXT_CONTENT_TYPES:
            return self.text_splitter.split_documents([document])
        return [document]

    def prepare_documents(
        self,
        documents: Sequence[Document],
    ) -> tuple[list[Document], list[str]]:
        """Split page text, preserve tables/images, and add stable metadata."""
        prepared: list[Document] = []
        point_ids: list[str] = []

        for document in documents:
            if not document.page_content.strip():
                continue

            for chunk_index, chunk in enumerate(self._split(document)):
                content = chunk.page_content.strip()
                if not content:
                    continue

                metadata = dict(chunk.metadata)
                source = str(metadata.get("source", "unknown"))
                content_type = str(metadata.get("content_type", "unknown"))
                page_number = metadata.get("page_number", -1)

                document_id = self._document_id(source)
                point_id = self._point_id(
                    source=source,
                    page_number=page_number,
                    content_type=content_type,
                    chunk_index=chunk_index,
                    table_index=metadata.get("table_index", ""),
                    image_index=metadata.get("image_index", ""),
                )

                metadata.update(
                    {
                        "document_id": document_id,
                        "filename": Path(source).name,
                        "chunk_index": chunk_index,
                        "chunk_id": point_id,
                    }
                )

                prepared.append(
                    Document(page_content=content, metadata=metadata)
                )
                point_ids.append(point_id)

        if not prepared:
            raise ValueError("No documents available for ingestion")

        return prepared, point_ids

    def _delete_existing_document(self, document_id: str) -> None:
        """Prevent stale/duplicate chunks when the same PDF is re-ingested."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def ingest_documents(
        self,
        documents: Sequence[Document],
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        try:
            prepared, point_ids = self.prepare_documents(documents)

            document_ids = sorted(
                {str(doc.metadata["document_id"]) for doc in prepared}
            )

            if replace_existing:
                for document_id in document_ids:
                    self._delete_existing_document(document_id)

            inserted_ids: list[str] = []

            for start in range(0, len(prepared), self.batch_size):
                end = start + self.batch_size
                inserted_ids.extend(
                    self.vector_store.add_documents(
                        documents=prepared[start:end],
                        ids=point_ids[start:end],
                    )
                )

            result = {
                "collection_name": self.collection_name,
                "source_document_count": len(document_ids),
                "input_document_count": len(documents),
                "indexed_document_count": len(prepared),
                "inserted_point_count": len(inserted_ids),
            }

            logger.info("document_ingestion_completed", **result)
            return result

        except Exception as exc:
            logger.exception("document_ingestion_failed", error=str(exc))
            raise DocumentPortalException(
                "Failed to ingest documents into Qdrant", exc
            ) from exc

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        tesseract_path: str | None = None,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        """Existing parser -> document preparation -> embeddings -> Qdrant."""
        try:
            parser = ComplexPDFParser(
                pdf_path=str(pdf_path),
                output_dir=str(output_dir),
                tesseract_path=tesseract_path,
            )
            parsed = parser.parse(save_output=True)

            result = self.ingest_documents(
                documents=parsed["documents"],
                replace_existing=replace_existing,
            )
            result.update(
                {
                    "pdf_path": str(pdf_path),
                    "parsed_pages": len(parsed["pages"]),
                    "parsed_tables": len(parsed["tables"]),
                    "parsed_images": len(parsed["images"]),
                }
            )
            return result

        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception(
                "pdf_ingestion_failed",
                pdf_path=str(pdf_path),
                error=str(exc),
            )
            raise DocumentPortalException(
                "Failed to parse and ingest PDF", exc
            ) from exc

    def get_vector_store(self) -> QdrantVectorStore:
        return self.vector_store


if __name__ == "__main__":
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "Client_Contracts_Policies_and_Incident_Records.pdf"
    )
    output_dir = PROJECT_ROOT / "data" / "parsed_pdf_output"

    ingestion = MultimodalDocumentIngestion()
    result = ingestion.ingest_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        tesseract_path=os.getenv("TESSERACT_PATH") or None,
        replace_existing=True,
    )

    print("Ingestion completed:", result)
