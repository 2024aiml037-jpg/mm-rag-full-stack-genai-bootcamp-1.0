from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt_library.prompt import (
    GENERATION_SMOKE_TEST_QUERY,
    MULTIMODAL_RAG_SYSTEM_PROMPT,
    format_multimodal_rag_user_prompt,
    format_retrieved_image_evidence_prompt,
)
from src.retriever import MultimodalQdrantRetriever


logger = CustomLogger().get_logger(__name__)

class MultimodalRAGGenerator:
    """
    Retrieve relevant MM-RAG content and generate a grounded answer.

    Current project behavior:
    - Text chunks are retrieved from Qdrant.
    - Tables are retrieved as their Markdown representation.
    - Image records are retrieved using their indexed OCR/text representation.
    - When an image record is retrieved, the original local image is also
      supplied to the multimodal model for final answer generation.

    Note:
    The current image retrieval quality depends on the text/OCR indexed for
    that image. A future enhancement can add VLM-generated image summaries or
    native image/multivector retrieval.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_images: int = 4,
        max_context_chars: int = 30_000,
        image_max_dimension: int = 1600,
    ) -> None:
        try:
            self.model_name = (
                model_name
                or os.getenv("OPENAI_CHAT_MODEL")
                or "gpt-4.1-mini"
            )

            self.temperature = temperature
            self.max_images = max_images
            self.max_context_chars = max_context_chars
            self.image_max_dimension = image_max_dimension

            self._validate_config()

            self.retriever = MultimodalQdrantRetriever(
                collection_name=collection_name
            )

            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
            )

            logger.info(
                "generation_initialized",
                model_name=self.model_name,
                collection_name=self.retriever.collection_name,
                max_images=self.max_images,
            )

        except Exception as exc:
            logger.exception(
                "generation_initialization_failed",
                error=str(exc),
            )
            raise DocumentPortalException(
                "Failed to initialize multimodal generation pipeline",
                exc,
            ) from exc

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing")

        if self.max_images < 0:
            raise ValueError("max_images cannot be negative")

        if self.max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be greater than 0"
            )

        if self.image_max_dimension <= 0:
            raise ValueError(
                "image_max_dimension must be greater than 0"
            )

    # -----------------------------------------------------------------
    # Image helpers
    # -----------------------------------------------------------------

    def _image_to_data_url(
        self,
        image_path: str | Path,
    ) -> str:
        """
        Convert a retrieved local image to an inline PNG data URL.

        Converting through Pillow makes the request more robust when the PDF
        contains embedded image formats that the model API may not accept
        directly.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Retrieved image does not exist: {path}"
            )

        with Image.open(path) as image:
            image.load()

            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")

            image.thumbnail(
                (
                    self.image_max_dimension,
                    self.image_max_dimension,
                )
            )

            buffer = BytesIO()
            image.save(
                buffer,
                format="PNG",
                optimize=True,
            )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    # -----------------------------------------------------------------
    # Context preparation
    # -----------------------------------------------------------------

    @staticmethod
    def _source_label(
        *,
        filename: str,
        page_number: Any,
    ) -> str:
        if page_number in {None, "", -1, "-1"}:
            return f"[{filename}]"

        return f"[{filename} p.{page_number}]"

    def _prepare_context(
        self,
        results: list[tuple[Any, float]],
    ) -> tuple[
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """
        Convert scored retrieval results into:
        1. Textual prompt context
        2. JSON-friendly source metadata
        3. Retrieved image descriptors
        """
        context_sections: list[str] = []
        sources: list[dict[str, Any]] = []
        image_candidates: list[dict[str, Any]] = []

        used_chars = 0
        seen_images: set[str] = set()

        for rank, (document, score) in enumerate(
            results,
            start=1,
        ):
            metadata = dict(document.metadata)

            filename = str(
                metadata.get("filename")
                or Path(
                    str(metadata.get("source", "unknown"))
                ).name
                or "unknown"
            )

            page_number = metadata.get(
                "page_number",
                "unknown",
            )

            content_type = str(
                metadata.get(
                    "content_type",
                    "unknown",
                )
            )

            source_label = self._source_label(
                filename=filename,
                page_number=page_number,
            )

            content = document.page_content.strip()

            header = (
                f"SOURCE {rank}\n"
                f"Citation: {source_label}\n"
                f"File: {filename}\n"
                f"Page: {page_number}\n"
                f"Content type: {content_type}\n"
                f"Retrieval score: {score:.4f}\n"
            )

            remaining = (
                self.max_context_chars
                - used_chars
                - len(header)
            )

            if remaining <= 0:
                break

            if len(content) > remaining:
                content = (
                    content[:remaining]
                    + "\n[Context truncated]"
                )

            section = (
                f"{header}"
                f"Content:\n{content}"
            )

            context_sections.append(section)
            used_chars += len(section)

            source_record = {
                "rank": rank,
                "score": float(score),
                "filename": filename,
                "page_number": page_number,
                "content_type": content_type,
                "citation": source_label,
                "document_id": metadata.get(
                    "document_id"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "table_index": metadata.get(
                    "table_index"
                ),
                "image_index": metadata.get(
                    "image_index"
                ),
                "image_path": metadata.get(
                    "image_path"
                ),
            }

            sources.append(source_record)

            image_path = metadata.get("image_path")

            if (
                content_type == "image"
                and image_path
                and len(image_candidates)
                < self.max_images
            ):
                normalized_path = str(
                    Path(str(image_path)).resolve()
                )

                if normalized_path not in seen_images:
                    seen_images.add(normalized_path)

                    image_candidates.append(
                        {
                            "source_rank": rank,
                            "citation": source_label,
                            "filename": filename,
                            "page_number": page_number,
                            "image_index": metadata.get(
                                "image_index"
                            ),
                            "image_path": str(
                                image_path
                            ),
                        }
                    )

        context = "\n\n---\n\n".join(
            context_sections
        )

        return context, sources, image_candidates

    # -----------------------------------------------------------------
    # Prompt / messages
    # -----------------------------------------------------------------

    def _build_messages(
        self,
        *,
        query: str,
        context: str,
        image_candidates: list[dict[str, Any]],
    ) -> tuple[
        list[Any],
        list[dict[str, Any]],
    ]:
        content_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": format_multimodal_rag_user_prompt(
                    query=query,
                    context=context,
                ),
            }
        ]

        used_images: list[dict[str, Any]] = []

        for image_info in image_candidates:
            try:
                data_url = self._image_to_data_url(
                    image_info["image_path"]
                )

                content_blocks.append(
                    {
                        "type": "text",
                        "text": format_retrieved_image_evidence_prompt(
                            citation=image_info["citation"],
                            image_index=image_info.get("image_index"),
                        ),
                    }
                )

                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    }
                )

                used_images.append(
                    dict(image_info)
                )

            except Exception as exc:
                logger.warning(
                    "retrieved_image_skipped",
                    image_path=image_info.get(
                        "image_path"
                    ),
                    error=str(exc),
                )

        messages = [
            SystemMessage(
                content=MULTIMODAL_RAG_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=content_blocks
            ),
        ]

        return messages, used_images

    # -----------------------------------------------------------------
    # Response helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        """
        Normalize LangChain AIMessage content into plain text.
        """
        text = getattr(response, "text", None)

        if isinstance(text, str) and text.strip():
            return text.strip()

        content = getattr(
            response,
            "content",
            "",
        )

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue

                if not isinstance(block, dict):
                    continue

                value = (
                    block.get("text")
                    or block.get("content")
                )

                if isinstance(value, str):
                    text_parts.append(value)

            return "\n".join(
                part for part in text_parts if part
            ).strip()

        return str(content).strip()

    @staticmethod
    def _usage_metadata(
        response: Any,
    ) -> dict[str, Any] | None:
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if isinstance(usage, dict):
            return usage

        return None

    # -----------------------------------------------------------------
    # Main generation pipeline
    # -----------------------------------------------------------------

    def generate(
        self,
        query: str,
        *,
        k: int = 6,
        min_score: float | None = None,
        filename: str | None = None,
        document_id: str | None = None,
        content_types: Iterable[str] | None = None,
        page_number: int | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve evidence and generate a grounded multimodal answer.

        Returns a JSON-friendly dictionary suitable for Streamlit/FastAPI.
        """
        try:
            query = query.strip()

            if not query:
                raise ValueError(
                    "Query cannot be empty"
                )

            if k <= 0:
                raise ValueError(
                    "k must be greater than 0"
                )

            results = (
                self.retriever.retrieve_with_scores(
                    query=query,
                    k=k,
                    filename=filename,
                    document_id=document_id,
                    content_types=content_types,
                    page_number=page_number,
                    page_from=page_from,
                    page_to=page_to,
                )
            )

            if min_score is not None:
                results = [
                    (document, score)
                    for document, score in results
                    if score >= min_score
                ]

            if not results:
                logger.info(
                    "generation_no_retrieval_results",
                    query=query,
                )

                return {
                    "query": query,
                    "answer": (
                        "I could not find enough relevant "
                        "information in the indexed documents "
                        "to answer this question."
                    ),
                    "sources": [],
                    "used_images": [],
                    "retrieval_count": 0,
                    "model_name": self.model_name,
                    "collection_name": (
                        self.retriever.collection_name
                    ),
                    "usage": None,
                }

            (
                context,
                sources,
                image_candidates,
            ) = self._prepare_context(results)

            messages, used_images = (
                self._build_messages(
                    query=query,
                    context=context,
                    image_candidates=image_candidates,
                )
            )

            logger.info(
                "generation_started",
                query=query,
                retrieved_documents=len(results),
                image_count=len(used_images),
                model_name=self.model_name,
            )

            response = self.llm.invoke(
                messages
            )

            answer = (
                self._extract_response_text(
                    response
                )
            )

            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "used_images": used_images,
                "retrieval_count": len(results),
                "model_name": self.model_name,
                "collection_name": (
                    self.retriever.collection_name
                ),
                "usage": self._usage_metadata(
                    response
                ),
            }

            logger.info(
                "generation_completed",
                query=query,
                retrieved_documents=len(results),
                used_images=len(used_images),
            )

            return result

        except Exception as exc:
            logger.exception(
                "generation_failed",
                query=query,
                error=str(exc),
            )
            raise DocumentPortalException(
                "Failed to generate MM-RAG answer",
                exc,
            ) from exc

    # Friendly alias for UI/API code.
    def answer_question(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.generate(
            query=query,
            **kwargs,
        )


# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    generator = MultimodalRAGGenerator()

    result = generator.answer_question(
        GENERATION_SMOKE_TEST_QUERY,
        k=6,
    )

    print("\nANSWER\n")
    print(result["answer"])

    print("\nSOURCES\n")
    for source in result["sources"]:
        print(
            source["citation"],
            "| type:",
            source["content_type"],
            "| score:",
            round(source["score"], 4),
        )

    if result["used_images"]:
        print("\nIMAGES SENT TO MODEL\n")
        for image in result["used_images"]:
            print(
                image["citation"],
                "->",
                image["image_path"],
            )
