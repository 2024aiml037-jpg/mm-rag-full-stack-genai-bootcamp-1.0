# -*- coding: utf-8 -*-
"""
Resume-aware ChatGPT-style Streamlit UI for the MM-RAG project.

Behavior:
1. Select an existing project PDF OR upload a new PDF.
2. If parsed artifacts already exist on disk, load them instead of parsing again.
3. If Qdrant already contains the document, reuse the existing index.
4. Only run missing stages.
5. Chat through MultimodalRAGGenerator.

Expected location:
    <project_root>/ui/app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from prompt_library.prompt import CHAT_SUGGESTED_PROMPTS
from src.parsing import ComplexPDFParser
from src.ingestion import MultimodalDocumentIngestion
from src.generation import MultimodalRAGGenerator


DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_ROOT = DATA_DIR / "parsed_pdf_output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_ROOT.mkdir(parents=True, exist_ok=True)

PARSED_JSON_FILES = (
    "page_records.json",
    "image_records.json",
    "table_records.json",
)

DEFAULT_COLLECTION = (
    os.getenv("QDRANT_COLLECTION_NAME")
    or "mm-rag-documents"
)


# ---------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="MM-RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        min-width: 330px;
        max-width: 380px;
    }

    .mmrag-title {
        font-size: 2.2rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin-bottom: 0.15rem;
    }

    .mmrag-subtitle {
        opacity: 0.72;
        margin-bottom: 1.2rem;
    }

    .source-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 10px 12px;
        margin: 6px 0;
    }

    .tiny-muted {
        opacity: 0.65;
        font-size: 0.82rem;
    }

    .ready-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.25);
        margin-right: 6px;
        margin-bottom: 8px;
        font-size: .83rem;
    }

    .resume-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 10px 12px;
        margin: 8px 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: 10px 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "document_token": None,
        "selected_pdf_path": None,
        "selected_pdf_fingerprint": None,
        "parsed_result": None,
        "parsed_pdf_path": None,
        "parsed_output_dir": None,
        "parsed_file_fingerprint": None,
        "parsed_restore_mode": None,
        "ingestion_result": None,
        "active_collection": None,
        "index_restore_mode": None,
        "index_check_token": None,
        "chat_messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state() -> None:
    """Clear all state tied to the previously selected document."""
    st.session_state.parsed_result = None
    st.session_state.parsed_pdf_path = None
    st.session_state.parsed_output_dir = None
    st.session_state.parsed_file_fingerprint = None
    st.session_state.parsed_restore_mode = None
    st.session_state.ingestion_result = None
    st.session_state.active_collection = None
    st.session_state.index_restore_mode = None
    st.session_state.index_check_token = None
    st.session_state.chat_messages = []


def clear_chat() -> None:
    st.session_state.chat_messages = []


init_session_state()


# ---------------------------------------------------------------------
# Cached external resources
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_qdrant_client(
    qdrant_url: str,
    qdrant_api_key: str | None,
) -> QdrantClient:
    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )


@st.cache_resource(show_spinner=False)
def get_generator(
    collection_name: str,
    model_name: str,
    max_images: int,
) -> MultimodalRAGGenerator:
    return MultimodalRAGGenerator(
        collection_name=collection_name,
        model_name=model_name,
        max_images=max_images,
    )


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def safe_filename(filename: str) -> str:
    return Path(filename).name


def file_fingerprint(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def fingerprint_file(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


def deepest_error_message(exc: BaseException) -> str:
    current: BaseException = exc
    visited: set[int] = set()

    while current.__cause__ is not None and id(current) not in visited:
        visited.add(id(current))
        current = current.__cause__

    return f"{type(current).__name__}: {current}"


def current_filename() -> str | None:
    path = st.session_state.selected_pdf_path

    if not path:
        return None

    return Path(path).name


def is_chat_ready() -> bool:
    return bool(
        st.session_state.ingestion_result
        and st.session_state.active_collection
    )


def discover_project_pdfs() -> list[Path]:
    """
    Find PDFs that already live in the project.

    We search:
    - data/*.pdf
    - data/uploads/*.pdf
    """
    discovered: dict[str, Path] = {}

    for folder in (DATA_DIR, UPLOAD_DIR):
        if not folder.exists():
            continue

        for path in folder.glob("*.pdf"):
            try:
                key = str(path.resolve()).lower()
            except OSError:
                key = str(path).lower()

            discovered[key] = path

    return sorted(
        discovered.values(),
        key=lambda path: path.name.lower(),
    )


# ---------------------------------------------------------------------
# Parsed-artifact persistence / restore
# ---------------------------------------------------------------------

def parsed_artifacts_complete(output_dir: Path) -> bool:
    return all(
        (output_dir / filename).exists()
        for filename in PARSED_JSON_FILES
    )


def parse_manifest_path(output_dir: Path) -> Path:
    return output_dir / "parse_manifest.json"


def write_parse_manifest(
    output_dir: Path,
    pdf_path: Path,
    fingerprint: str,
) -> None:
    manifest = {
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "sha256": fingerprint,
    }

    parse_manifest_path(output_dir).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def read_parse_manifest(
    output_dir: Path,
) -> dict[str, Any] | None:
    manifest_path = parse_manifest_path(
        output_dir
    )

    if not manifest_path.exists():
        return None

    try:
        return json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def legacy_output_mentions_pdf(
    output_dir: Path,
    filename: str,
) -> bool:
    """
    Old terminal parsing did not create a manifest.

    rag_ready_documents.md contains metadata with the original source path,
    therefore the filename can be used to identify that legacy output.
    """
    rag_file = output_dir / "rag_ready_documents.md"

    if not rag_file.exists():
        return False

    try:
        content = rag_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        return filename.lower() in content.lower()
    except Exception:
        return False


def find_existing_parsed_output(
    pdf_path: Path,
    fingerprint: str,
) -> tuple[Path | None, str | None]:
    """
    Detect both supported layouts.

    New UI layout:
        data/parsed_pdf_output/<pdf-stem>/

    Old terminal layout used earlier in this project:
        data/parsed_pdf_output/
    """
    candidates = [
        PARSED_ROOT / pdf_path.stem,
        PARSED_ROOT,
    ]

    for output_dir in candidates:
        if not parsed_artifacts_complete(
            output_dir
        ):
            continue

        manifest = read_parse_manifest(
            output_dir
        )

        if manifest:
            stored_hash = str(
                manifest.get("sha256", "")
            )

            if stored_hash == fingerprint:
                return output_dir, "sha256"

            # A manifest exists and says this is a different file/version.
            continue

        # No manifest: support outputs generated by the existing terminal
        # pipeline. Per-document folders are reasonably attributable by stem.
        if output_dir != PARSED_ROOT:
            if (
                output_dir.name.lower()
                == pdf_path.stem.lower()
            ):
                return output_dir, "legacy-folder"

        # Old ingestion.py passed data/parsed_pdf_output directly.
        if legacy_output_mentions_pdf(
            output_dir,
            pdf_path.name,
        ):
            return output_dir, "legacy-filename"

    return None, None


def load_existing_parsed_result(
    pdf_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Rebuild the parser's in-memory result from JSON artifacts.

    No PDF text extraction, OCR, image extraction, or table extraction runs.
    create_langchain_documents() only rebuilds LangChain Document objects from
    the already-saved records.
    """
    with (
        output_dir / "page_records.json"
    ).open("r", encoding="utf-8") as file:
        pages = json.load(file)

    with (
        output_dir / "image_records.json"
    ).open("r", encoding="utf-8") as file:
        images = json.load(file)

    with (
        output_dir / "table_records.json"
    ).open("r", encoding="utf-8") as file:
        tables = json.load(file)

    parser = ComplexPDFParser(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        tesseract_path=(
            os.getenv("TESSERACT_PATH")
            or None
        ),
    )

    parser.page_records = pages
    parser.image_records = images
    parser.table_records = tables

    documents = (
        parser.create_langchain_documents()
    )

    return {
        "pages": pages,
        "images": images,
        "tables": tables,
        "documents": documents,
        "output_dir": str(output_dir),
    }


def try_restore_parsed_state(
    pdf_path: Path,
    fingerprint: str,
) -> bool:
    output_dir, mode = (
        find_existing_parsed_output(
            pdf_path=pdf_path,
            fingerprint=fingerprint,
        )
    )

    if output_dir is None:
        return False

    parsed = load_existing_parsed_result(
        pdf_path=pdf_path,
        output_dir=output_dir,
    )

    st.session_state.parsed_result = parsed
    st.session_state.parsed_pdf_path = str(
        pdf_path
    )
    st.session_state.parsed_output_dir = str(
        output_dir
    )
    st.session_state.parsed_file_fingerprint = (
        fingerprint
    )
    st.session_state.parsed_restore_mode = (
        mode
    )

    return True


# ---------------------------------------------------------------------
# Qdrant existing-index detection
# ---------------------------------------------------------------------

def qdrant_document_filter(
    *,
    field: str,
    value: str,
) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=field,
                match=models.MatchValue(
                    value=value
                ),
            )
        ]
    )


def qdrant_has_points(
    client: QdrantClient,
    collection_name: str,
    query_filter: models.Filter,
) -> bool:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    return bool(points)


def find_existing_qdrant_index(
    *,
    collection_name: str,
    filename: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """
    Check whether this PDF has already been indexed.

    New UI ingestions store metadata.file_sha256, so future runs can match the
    exact PDF bytes.

    Old terminal-created points do not have file_sha256, therefore we fall back
    to metadata.filename for backward compatibility.
    """
    qdrant_url = (
        os.getenv("QDRANT_URL")
        or os.getenv(
            "QDRANT_Cluster_Endpoint"
        )
    )
    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY"
    )

    if not qdrant_url:
        return None

    client = get_qdrant_client(
        qdrant_url,
        qdrant_api_key,
    )

    if not client.collection_exists(
        collection_name
    ):
        return None

    # Best match: exact content hash.
    exact_filter = qdrant_document_filter(
        field="metadata.file_sha256",
        value=fingerprint,
    )

    try:
        exact_hash_match = qdrant_has_points(
            client,
            collection_name,
            exact_filter,
        )
    except Exception:
        # Older collections may not have a payload index for file_sha256
        # (and strict-mode deployments can reject unindexed filtering).
        exact_hash_match = False

    if exact_hash_match:
        info = client.get_collection(
            collection_name
        )

        return {
            "collection_name": collection_name,
            "loaded_existing": True,
            "match_mode": "sha256",
            "collection_points": getattr(
                info,
                "points_count",
                None,
            ),
        }

    # Backward compatibility with points created before SHA metadata was added.
    legacy_filter = qdrant_document_filter(
        field="metadata.filename",
        value=filename,
    )

    if qdrant_has_points(
        client,
        collection_name,
        legacy_filter,
    ):
        info = client.get_collection(
            collection_name
        )

        return {
            "collection_name": collection_name,
            "loaded_existing": True,
            "match_mode": "legacy-filename",
            "collection_points": getattr(
                info,
                "points_count",
                None,
            ),
        }

    return None


def try_restore_index_state(
    *,
    collection_name: str,
    filename: str,
    fingerprint: str,
) -> bool:
    existing = find_existing_qdrant_index(
        collection_name=collection_name,
        filename=filename,
        fingerprint=fingerprint,
    )

    if not existing:
        return False

    st.session_state.ingestion_result = (
        existing
    )
    st.session_state.active_collection = (
        collection_name
    )
    st.session_state.index_restore_mode = (
        existing.get("match_mode")
    )

    return True


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------

def render_config_status() -> None:
    openai_key = bool(
        os.getenv("OPENAI_API_KEY")
    )
    qdrant_url = (
        os.getenv("QDRANT_URL")
        or os.getenv(
            "QDRANT_Cluster_Endpoint"
        )
    )
    qdrant_key = bool(
        os.getenv("QDRANT_API_KEY")
    )
    tesseract_path = os.getenv(
        "TESSERACT_PATH"
    )

    with st.sidebar.expander(
        "⚙️ Environment",
        expanded=False,
    ):
        st.write(
            f"{'✅' if openai_key else '❌'} OpenAI API key"
        )
        st.write(
            f"{'✅' if qdrant_url else '❌'} Qdrant URL"
        )
        st.write(
            f"{'✅' if qdrant_key else '⚠️'} Qdrant API key"
        )
        st.write(
            f"{'✅' if tesseract_path else 'ℹ️'} Tesseract path"
        )


def render_pipeline_badges() -> None:
    parsed = (
        st.session_state.parsed_result
        is not None
    )
    indexed = (
        st.session_state.ingestion_result
        is not None
    )
    ready = is_chat_ready()

    st.markdown(
        (
            f'<span class="ready-pill">'
            f'{"✅" if parsed else "○"} Parsed'
            f'</span>'
            f'<span class="ready-pill">'
            f'{"✅" if indexed else "○"} Indexed'
            f'</span>'
            f'<span class="ready-pill">'
            f'{"✅" if ready else "○"} Chat ready'
            f'</span>'
        ),
        unsafe_allow_html=True,
    )


def render_parse_summary(
    parsed: dict[str, Any],
) -> None:
    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Pages",
        len(parsed.get("pages", [])),
    )
    col2.metric(
        "Tables",
        len(parsed.get("tables", [])),
    )
    col3.metric(
        "Images",
        len(parsed.get("images", [])),
    )
    col4.metric(
        "Documents",
        len(parsed.get("documents", [])),
    )


def render_document_inspector(
    parsed: dict[str, Any],
) -> None:
    with st.expander(
        "🔎 Document inspector",
        expanded=False,
    ):
        text_tab, table_tab, image_tab = (
            st.tabs(
                [
                    "Text / OCR",
                    "Tables",
                    "Images",
                ]
            )
        )

        with text_tab:
            page_documents = [
                document
                for document in parsed.get(
                    "documents",
                    [],
                )
                if document.metadata.get(
                    "content_type"
                )
                == "page_text_plus_ocr"
            ]

            if not page_documents:
                st.info(
                    "No page text documents were produced."
                )
            else:
                page_options = [
                    int(
                        document.metadata.get(
                            "page_number",
                            index + 1,
                        )
                    )
                    for index, document
                    in enumerate(
                        page_documents
                    )
                ]

                selected_page = st.selectbox(
                    "Preview page",
                    options=page_options,
                    key="preview_page",
                )

                selected_document = next(
                    (
                        document
                        for document
                        in page_documents
                        if int(
                            document.metadata.get(
                                "page_number",
                                -1,
                            )
                        )
                        == selected_page
                    ),
                    None,
                )

                if selected_document:
                    st.text(
                        selected_document
                        .page_content[:12000]
                    )

        with table_tab:
            tables = parsed.get(
                "tables",
                [],
            )

            if not tables:
                st.info(
                    "No tables detected."
                )
            else:
                labels = [
                    (
                        f"Page "
                        f"{table.get('page_number', '?')}"
                        f" · Table "
                        f"{table.get('table_index', '?')}"
                    )
                    for table in tables
                ]

                index = st.selectbox(
                    "Preview table",
                    options=range(
                        len(tables)
                    ),
                    format_func=lambda i: (
                        labels[i]
                    ),
                    key="preview_table",
                )

                table = tables[index]

                if table.get("markdown"):
                    st.markdown(
                        table["markdown"]
                    )
                else:
                    st.write(
                        table.get(
                            "raw_table"
                        )
                    )

        with image_tab:
            images = parsed.get(
                "images",
                [],
            )

            if not images:
                st.info(
                    "No embedded images detected."
                )
            else:
                labels = [
                    (
                        f"Page "
                        f"{image.get('page_number', '?')}"
                        f" · Image "
                        f"{image.get('image_index', '?')}"
                    )
                    for image in images
                ]

                index = st.selectbox(
                    "Preview image",
                    options=range(
                        len(images)
                    ),
                    format_func=lambda i: (
                        labels[i]
                    ),
                    key="preview_image",
                )

                image_record = images[index]
                image_path = Path(
                    str(
                        image_record.get(
                            "image_path",
                            "",
                        )
                    )
                )

                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=labels[index],
                        width="stretch",
                    )

                    ocr_text = str(
                        image_record.get(
                            "image_ocr_text",
                            "",
                        )
                    ).strip()

                    if ocr_text:
                        with st.expander(
                            "OCR text"
                        ):
                            st.text(
                                ocr_text[:6000]
                            )
                else:
                    st.warning(
                        f"Image file not found: "
                        f"{image_path}"
                    )


def render_index_status() -> None:
    result = (
        st.session_state.ingestion_result
        or {}
    )

    if result.get("loaded_existing"):
        mode = result.get(
            "match_mode"
        )

        st.success(
            "Existing Qdrant index loaded — "
            "no re-ingestion was performed."
        )

        if mode == "sha256":
            st.caption(
                "Matched using exact PDF SHA-256."
            )
        else:
            st.caption(
                "Matched using filename because this "
                "index was created by the older terminal "
                "pipeline before SHA metadata was stored."
            )

        points = result.get(
            "collection_points"
        )

        if points is not None:
            st.metric(
                "Collection points",
                points,
            )

        st.caption(
            f"Collection: "
            f"{result.get('collection_name')}"
        )

    else:
        st.success(
            "Document indexed successfully."
        )

        col1, col2, col3 = (
            st.columns(3)
        )
        col1.metric(
            "Parsed docs",
            result.get(
                "input_document_count",
                0,
            ),
        )
        col2.metric(
            "Indexed chunks",
            result.get(
                "indexed_document_count",
                0,
            ),
        )
        col3.metric(
            "Qdrant points",
            result.get(
                "inserted_point_count",
                0,
            ),
        )



def render_sidebar_document_workspace(
    selected_pdf_path: Path | None,
) -> None:
    """
    Keep document/pipeline details out of the main chat canvas.

    Everything related to parsing, restored artifacts, document statistics,
    inspection, and Qdrant index state is shown in the sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Document status")

    # Pipeline badges belong in the workspace sidebar, not the chat canvas.
    with st.sidebar:
        render_pipeline_badges()

    if selected_pdf_path is None:
        st.sidebar.caption("No document selected.")
        return

    st.sidebar.caption(selected_pdf_path.name)

    with st.sidebar.expander(
        "📄 Document details",
        expanded=False,
    ):
        st.caption(str(selected_pdf_path))

        parsed = st.session_state.parsed_result

        if parsed:
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            row1_col1.metric(
                "Pages",
                len(parsed.get("pages", [])),
            )
            row1_col2.metric(
                "Tables",
                len(parsed.get("tables", [])),
            )
            row2_col1.metric(
                "Images",
                len(parsed.get("images", [])),
            )
            row2_col2.metric(
                "Documents",
                len(parsed.get("documents", [])),
            )

            if st.session_state.parsed_restore_mode:
                st.success(
                    "Existing parsed data restored. "
                    "OCR/table/image extraction was skipped."
                )
            else:
                st.success(
                    "PDF parsed in this app session."
                )
        else:
            st.info(
                "No reusable parsed data found yet."
            )

    if st.session_state.parsed_result:
        # Existing inspector uses normal Streamlit primitives. Running it
        # inside the sidebar context keeps all of its tabs/selectors there.
        with st.sidebar:
            render_document_inspector(
                st.session_state.parsed_result
            )

    with st.sidebar.expander(
        "🗄️ Qdrant index status",
        expanded=False,
    ):
        result = (
            st.session_state.ingestion_result
            or {}
        )

        if not result:
            st.info(
                "No reusable Qdrant index found yet."
            )
            return

        collection = (
            result.get("collection_name")
            or st.session_state.active_collection
            or "unknown"
        )

        if result.get("loaded_existing"):
            mode = result.get("match_mode")

            st.success(
                "Existing Qdrant index loaded. "
                "Embeddings were not created again."
            )

            if mode == "sha256":
                st.caption(
                    "Matched using exact PDF SHA-256."
                )
            else:
                st.caption(
                    "Matched using filename from the "
                    "older terminal-created index."
                )

            points = result.get(
                "collection_points"
            )

            if points is not None:
                st.metric(
                    "Collection points",
                    points,
                )

        else:
            st.success(
                "Document indexed in this app session."
            )

            col1, col2 = st.columns(2)
            col1.metric(
                "Chunks",
                result.get(
                    "indexed_document_count",
                    0,
                ),
            )
            col2.metric(
                "Points",
                result.get(
                    "inserted_point_count",
                    0,
                ),
            )

        st.caption(
            f"Collection: {collection}"
        )

def render_source_cards(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    with st.expander(
        f"Sources · {len(sources)} retrieved",
        expanded=False,
    ):
        for source in sources:
            citation = source.get(
                "citation",
                "Unknown source",
            )
            content_type = source.get(
                "content_type",
                "unknown",
            )
            score = source.get("score")

            score_text = (
                f"{float(score):.4f}"
                if isinstance(
                    score,
                    (float, int),
                )
                else "n/a"
            )

            st.markdown(
                f"""
<div class="source-card">
    <strong>{citation}</strong><br/>
    <span class="tiny-muted">
        type: {content_type} · similarity: {score_text}
    </span>
</div>
""",
                unsafe_allow_html=True,
            )


def render_used_images(
    images: list[dict[str, Any]],
) -> None:
    valid_images = [
        image
        for image in images
        if image.get("image_path")
        and Path(
            str(image["image_path"])
        ).exists()
    ]

    if not valid_images:
        return

    with st.expander(
        (
            "Retrieved visual evidence · "
            f"{len(valid_images)}"
        ),
        expanded=False,
    ):
        columns = st.columns(
            min(3, len(valid_images))
        )

        for index, image in enumerate(
            valid_images
        ):
            with columns[
                index % len(columns)
            ]:
                st.image(
                    str(image["image_path"]),
                    caption=image.get(
                        "citation",
                        "Retrieved image",
                    ),
                    width="stretch",
                )


def render_assistant_message(
    message: dict[str, Any],
) -> None:
    st.markdown(
        message.get("content", "")
    )

    render_source_cards(
        message.get("sources", [])
    )
    render_used_images(
        message.get("used_images", [])
    )

    metadata = message.get(
        "metadata",
        {},
    )

    details: list[str] = []

    if metadata.get("model_name"):
        details.append(
            str(metadata["model_name"])
        )

    if (
        metadata.get(
            "retrieval_count"
        )
        is not None
    ):
        details.append(
            f"{metadata['retrieval_count']} retrieved"
        )

    if details:
        st.caption(
            " · ".join(details)
        )


def render_chat_history() -> None:
    for message in (
        st.session_state.chat_messages
    ):
        role = message.get(
            "role",
            "assistant",
        )
        avatar = (
            "🧑‍💻"
            if role == "user"
            else "🧠"
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):
            if role == "assistant":
                render_assistant_message(
                    message
                )
            else:
                st.markdown(
                    message.get(
                        "content",
                        "",
                    )
                )


# ---------------------------------------------------------------------
# Sidebar: select document
# ---------------------------------------------------------------------

st.sidebar.markdown("## 🧠 MM-RAG")
st.sidebar.caption(
    "Resume-aware document workspace"
)
render_config_status()

existing_pdfs = discover_project_pdfs()

source_options = ["Upload new PDF"]

if existing_pdfs:
    source_options.insert(
        0,
        "Use existing project PDF",
    )

source_mode = st.sidebar.radio(
    "Document source",
    options=source_options,
)

selected_pdf_path: Path | None = None
selected_fingerprint: str | None = None

if source_mode == "Use existing project PDF":
    selected_index = st.sidebar.selectbox(
        "Existing PDF",
        options=range(
            len(existing_pdfs)
        ),
        format_func=lambda i: (
            existing_pdfs[i].name
        ),
    )

    selected_pdf_path = (
        existing_pdfs[selected_index]
    )

    try:
        selected_fingerprint = (
            fingerprint_file(
                selected_pdf_path
            )
        )
    except Exception as exc:
        st.sidebar.error(
            deepest_error_message(exc)
        )

else:
    uploaded_file = (
        st.sidebar.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            key="pdf_uploader",
        )
    )

    if uploaded_file is not None:
        uploaded_bytes = (
            uploaded_file.getvalue()
        )
        filename = safe_filename(
            uploaded_file.name
        )
        selected_fingerprint = (
            file_fingerprint(
                uploaded_bytes
            )
        )

        selected_pdf_path = (
            UPLOAD_DIR / filename
        )

        # Persist uploader bytes because Streamlit upload data itself is
        # session memory, while our parser/image paths need stable files.
        if (
            not selected_pdf_path.exists()
            or fingerprint_file(
                selected_pdf_path
            )
            != selected_fingerprint
        ):
            selected_pdf_path.write_bytes(
                uploaded_bytes
            )

        st.sidebar.caption(
            f"{filename} · "
            f"{len(uploaded_bytes) / (1024 * 1024):.2f} MB"
        )


# ---------------------------------------------------------------------
# Document changed -> restore local parsed artifacts
# ---------------------------------------------------------------------

if (
    selected_pdf_path is not None
    and selected_fingerprint is not None
):
    document_token = (
        f"{selected_pdf_path.resolve()}|"
        f"{selected_fingerprint}"
    )

    if (
        st.session_state.document_token
        != document_token
    ):
        reset_document_state()

        st.session_state.document_token = (
            document_token
        )
        st.session_state.selected_pdf_path = str(
            selected_pdf_path
        )
        st.session_state.selected_pdf_fingerprint = (
            selected_fingerprint
        )

        try:
            try_restore_parsed_state(
                pdf_path=selected_pdf_path,
                fingerprint=selected_fingerprint,
            )
        except Exception as exc:
            # Restore failure should not stop the user from re-parsing.
            st.sidebar.warning(
                "Saved parsed artifacts were found "
                "but could not be loaded."
            )
            st.sidebar.code(
                deepest_error_message(exc)
            )

    else:
        st.session_state.selected_pdf_path = str(
            selected_pdf_path
        )
        st.session_state.selected_pdf_fingerprint = (
            selected_fingerprint
        )

else:
    if (
        st.session_state.document_token
        is not None
    ):
        st.session_state.document_token = None
        reset_document_state()


# ---------------------------------------------------------------------
# Sidebar: parsing
# ---------------------------------------------------------------------

if selected_pdf_path is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Document pipeline")

    if (
        st.session_state.parsed_result
        is not None
    ):
        if (
            st.session_state.parsed_restore_mode
            is not None
        ):
            st.sidebar.success(
                "1 · Existing parsed data loaded"
            )
            st.sidebar.caption(
                "No OCR/parsing rerun."
            )
        else:
            st.sidebar.success(
                "1 · PDF parsed"
            )

        force_parse = st.sidebar.button(
            "Re-parse PDF",
            use_container_width=True,
        )
    else:
        force_parse = st.sidebar.button(
            "1 · Parse PDF",
            type="primary",
            use_container_width=True,
        )

    if force_parse:
        try:
            output_dir = (
                PARSED_ROOT
                / selected_pdf_path.stem
            )

            if output_dir.exists():
                shutil.rmtree(
                    output_dir
                )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with st.status(
                "Parsing PDF...",
                expanded=True,
            ) as status:
                parser = ComplexPDFParser(
                    pdf_path=str(
                        selected_pdf_path
                    ),
                    output_dir=str(
                        output_dir
                    ),
                    tesseract_path=(
                        os.getenv(
                            "TESSERACT_PATH"
                        )
                        or None
                    ),
                )

                parsed = parser.parse(
                    save_output=True
                )

                write_parse_manifest(
                    output_dir=output_dir,
                    pdf_path=(
                        selected_pdf_path
                    ),
                    fingerprint=(
                        selected_fingerprint
                    ),
                )

                st.session_state.parsed_result = (
                    parsed
                )
                st.session_state.parsed_pdf_path = str(
                    selected_pdf_path
                )
                st.session_state.parsed_output_dir = str(
                    output_dir
                )
                st.session_state.parsed_file_fingerprint = (
                    selected_fingerprint
                )
                st.session_state.parsed_restore_mode = (
                    None
                )

                # Parsed bytes have changed/rebuilt, so re-check the index.
                st.session_state.ingestion_result = (
                    None
                )
                st.session_state.active_collection = (
                    None
                )
                st.session_state.index_restore_mode = (
                    None
                )
                st.session_state.index_check_token = (
                    None
                )
                st.session_state.chat_messages = []

                status.update(
                    label="PDF parsed",
                    state="complete",
                    expanded=False,
                )

            st.rerun()

        except Exception as exc:
            st.sidebar.error(
                "PDF parsing failed."
            )
            st.sidebar.code(
                deepest_error_message(exc)
            )


# ---------------------------------------------------------------------
# Sidebar: collection + existing-index restore
# ---------------------------------------------------------------------

collection_name = DEFAULT_COLLECTION

if selected_pdf_path is not None:
    collection_name = (
        st.sidebar.text_input(
            "Qdrant collection",
            value=DEFAULT_COLLECTION,
            key="collection_name_input",
        )
        .strip()
    )

    index_check_token = (
        f"{st.session_state.document_token}|"
        f"{collection_name}"
    )

    if (
        collection_name
        and st.session_state.index_check_token
        != index_check_token
    ):
        # Changing collection must invalidate the previously loaded index state.
        st.session_state.ingestion_result = (
            None
        )
        st.session_state.active_collection = (
            None
        )
        st.session_state.index_restore_mode = (
            None
        )

        try:
            try_restore_index_state(
                collection_name=(
                    collection_name
                ),
                filename=(
                    selected_pdf_path.name
                ),
                fingerprint=(
                    selected_fingerprint
                ),
            )
        except Exception as exc:
            # Qdrant being temporarily unavailable should not prevent local
            # parse inspection.
            st.sidebar.warning(
                "Could not check the existing Qdrant index."
            )
            st.sidebar.code(
                deepest_error_message(exc)
            )

        st.session_state.index_check_token = (
            index_check_token
        )


# ---------------------------------------------------------------------
# Sidebar: ingestion only when index missing
# ---------------------------------------------------------------------

if (
    selected_pdf_path is not None
    and st.session_state.parsed_result
    is not None
):
    if (
        st.session_state.ingestion_result
        is not None
    ):
        if (
            st.session_state.index_restore_mode
            is not None
        ):
            st.sidebar.success(
                "2 · Existing Qdrant index loaded"
            )
            st.sidebar.caption(
                "No embeddings/re-ingestion rerun."
            )
        else:
            st.sidebar.success(
                "2 · Indexed in Qdrant"
            )

        if st.sidebar.button(
            "Re-index document",
            use_container_width=True,
        ):
            st.session_state.ingestion_result = (
                None
            )
            st.session_state.active_collection = (
                None
            )
            st.session_state.index_restore_mode = (
                None
            )
            st.session_state.chat_messages = []
            get_generator.clear()
            st.rerun()

    else:
        with st.sidebar.expander(
            "Advanced indexing",
            expanded=False,
        ):
            chunk_size = st.number_input(
                "Chunk size",
                min_value=200,
                max_value=10000,
                value=2000,
                step=100,
                key="chunk_size_input",
            )

            chunk_overlap = (
                st.number_input(
                    "Chunk overlap",
                    min_value=0,
                    max_value=2000,
                    value=120,
                    step=20,
                    key="chunk_overlap_input",
                )
            )

            replace_existing = (
                st.checkbox(
                    "Replace existing document points",
                    value=True,
                    key="replace_existing_input",
                )
            )

        invalid_chunking = (
            chunk_overlap
            >= chunk_size
        )

        if invalid_chunking:
            st.sidebar.error(
                "Chunk overlap must be smaller than chunk size."
            )

        if st.sidebar.button(
            "2 · Ingest to Qdrant",
            type="primary",
            use_container_width=True,
            disabled=(
                invalid_chunking
                or not collection_name
            ),
        ):
            try:
                documents = (
                    st.session_state
                    .parsed_result["documents"]
                )

                # Store exact file identity in future Qdrant points.
                for document in documents:
                    document.metadata[
                        "file_sha256"
                    ] = (
                        selected_fingerprint
                    )

                with st.status(
                    "Indexing document...",
                    expanded=True,
                ) as status:
                    ingestion = (
                        MultimodalDocumentIngestion(
                            collection_name=(
                                collection_name
                            ),
                            chunk_size=int(
                                chunk_size
                            ),
                            chunk_overlap=int(
                                chunk_overlap
                            ),
                        )
                    )

                    result = (
                        ingestion.ingest_documents(
                            documents=documents,
                            replace_existing=(
                                replace_existing
                            ),
                        )
                    )

                    result[
                        "file_sha256"
                    ] = selected_fingerprint
                    result[
                        "loaded_existing"
                    ] = False

                    st.session_state.ingestion_result = (
                        result
                    )
                    st.session_state.active_collection = (
                        collection_name
                    )
                    st.session_state.index_restore_mode = (
                        None
                    )
                    st.session_state.chat_messages = []

                    get_generator.clear()

                    status.update(
                        label=(
                            "Qdrant indexing complete"
                        ),
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except Exception as exc:
                st.sidebar.error(
                    "Qdrant ingestion failed."
                )
                st.sidebar.code(
                    deepest_error_message(
                        exc
                    )
                )


# ---------------------------------------------------------------------
# Sidebar: chat settings
# ---------------------------------------------------------------------

if is_chat_ready():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "#### Chat settings"
    )

    chat_model = (
        st.sidebar.text_input(
            "Model",
            value=(
                os.getenv(
                    "OPENAI_CHAT_MODEL"
                )
                or "gpt-4.1-mini"
            ),
            key="chat_model_input",
        )
    )

    top_k = st.sidebar.slider(
        "Top-K retrieval",
        min_value=2,
        max_value=15,
        value=6,
        key="top_k_input",
    )

    max_images = st.sidebar.slider(
        "Max retrieved images",
        min_value=0,
        max_value=8,
        value=4,
        key="max_images_input",
    )

    current_pdf_only = (
        st.sidebar.checkbox(
            "Search only current PDF",
            value=True,
            key="current_pdf_only_input",
        )
    )

    if st.sidebar.button(
        "Clear chat",
        use_container_width=True,
    ):
        clear_chat()
        st.rerun()



# ---------------------------------------------------------------------
# Sidebar: document / parsing / index information
# ---------------------------------------------------------------------

render_sidebar_document_workspace(
    selected_pdf_path
)


# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------

st.markdown(
    (
        '<div class="mmrag-title">'
        "Multimodal Document Assistant"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown(
    (
        '<div class="mmrag-subtitle">'
        "Ask grounded questions across text, OCR, tables, "
        "and retrieved visual evidence."
        "</div>"
    ),
    unsafe_allow_html=True,
)



# ---------------------------------------------------------------------
# Main chat readiness
# ---------------------------------------------------------------------

if selected_pdf_path is None:
    st.markdown("### Start a document conversation")
    st.caption(
        "Choose an existing PDF or upload a new one from the sidebar."
    )
    st.stop()

if st.session_state.parsed_result is None:
    st.markdown("### Preparing your document")
    st.caption(
        "No reusable parsed artifacts were found. "
        "Use **Parse PDF** in the sidebar."
    )
    st.stop()

if st.session_state.ingestion_result is None:
    st.markdown("### Almost ready")
    st.caption(
        "The PDF is parsed. Use **Ingest to Qdrant** in the sidebar "
        "to enable document chat."
    )
    st.stop()


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

if not st.session_state.chat_messages:
    st.markdown(
        "### What would you like to know?"
    )
    st.caption(
        "Ask about policies, contracts, incidents, tables, charts, or diagrams."
    )

    suggestions = CHAT_SUGGESTED_PROMPTS

    columns = st.columns(2)
    suggested_prompt: str | None = None

    for index, suggestion in enumerate(
        suggestions
    ):
        with columns[index % 2]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                suggested_prompt = (
                    suggestion
                )
else:
    suggested_prompt = None


render_chat_history()

typed_prompt = st.chat_input(
    "Ask anything about the document...",
    key="document_chat_input",
)

prompt = suggested_prompt or typed_prompt

if prompt:
    user_message = {
        "role": "user",
        "content": str(prompt),
    }

    st.session_state.chat_messages.append(
        user_message
    )

    with st.chat_message(
        "user",
        avatar="🧑‍💻",
    ):
        st.markdown(str(prompt))

    with st.chat_message(
        "assistant",
        avatar="🧠",
    ):
        try:
            with st.status(
                "Searching the document...",
                expanded=False,
            ) as status:
                generator = get_generator(
                    collection_name=(
                        st.session_state
                        .active_collection
                    ),
                    model_name=(
                        chat_model.strip()
                    ),
                    max_images=int(
                        max_images
                    ),
                )

                filename_filter = (
                    current_filename()
                    if current_pdf_only
                    else None
                )

                result = (
                    generator
                    .answer_question(
                        query=str(prompt),
                        k=int(top_k),
                        filename=(
                            filename_filter
                        ),
                    )
                )

                status.update(
                    label="Answer ready",
                    state="complete",
                    expanded=False,
                )

            assistant_message = {
                "role": "assistant",
                "content": result.get(
                    "answer",
                    "No answer was generated.",
                ),
                "sources": result.get(
                    "sources",
                    [],
                ),
                "used_images": result.get(
                    "used_images",
                    [],
                ),
                "metadata": {
                    "model_name": (
                        result.get(
                            "model_name"
                        )
                    ),
                    "retrieval_count": (
                        result.get(
                            "retrieval_count"
                        )
                    ),
                    "usage": result.get(
                        "usage"
                    ),
                },
            }

            st.session_state.chat_messages.append(
                assistant_message
            )

            render_assistant_message(
                assistant_message
            )

        except Exception as exc:
            error_text = (
                deepest_error_message(
                    exc
                )
            )

            st.error(
                "I could not generate an answer."
            )
            st.code(error_text)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The RAG pipeline returned an error: "
                        f"`{error_text}`"
                    ),
                    "sources": [],
                    "used_images": [],
                    "metadata": {},
                }
            )
