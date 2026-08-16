"""Central prompt library for the multimodal RAG application."""

from __future__ import annotations


MULTIMODAL_RAG_SYSTEM_PROMPT = """
You are the answer-generation component of a multimodal RAG system.

Rules:
1. Answer only from the retrieved context and retrieved images supplied to you.
2. Do not use outside knowledge to fill missing facts.
3. If the retrieved evidence is insufficient, clearly say that the available
   context does not contain enough information.
4. Treat text, OCR, tables, and images as evidence. If sources disagree,
   mention the disagreement rather than inventing a reconciliation.
5. Cite factual claims using the exact citation labels supplied in the
   retrieved context, for example: [report.pdf p.4].
6. Prefer the most directly relevant source. Do not cite unrelated sources.
7. Keep numerical values, dates, names, policy terms, and incident details
   faithful to the retrieved evidence.
8. If an image is supplied, inspect the actual image; do not rely only on its
   OCR text.
9. Do not mention vector databases, retrieval scores, embeddings, prompts, or
   internal RAG mechanics in the final answer unless the user explicitly asks.
""".strip()


MULTIMODAL_RAG_USER_PROMPT = (
    "USER QUESTION:\n{query}\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "Answer the user question using only the evidence above "
    "and any retrieved images attached below."
)


RETRIEVED_IMAGE_EVIDENCE_PROMPT = (
    "\nRetrieved image evidence:\n"
    "Citation: {citation}\n"
    "Image index: {image_index}"
)


CHAT_SUGGESTED_PROMPTS = (
    "Summarize the key policies in this document.",
    "What are the most important incident details?",
    "What important information appears in the tables?",
    "Explain useful visual or diagram evidence.",
)


GENERATION_SMOKE_TEST_QUERY = (
    "Summarize the most important client policy "
    "or incident information in the document."
)


def format_multimodal_rag_user_prompt(
    *,
    query: str,
    context: str,
) -> str:
    """Build the grounded user prompt from the current query and context."""
    return MULTIMODAL_RAG_USER_PROMPT.format(
        query=query,
        context=context,
    )


def format_retrieved_image_evidence_prompt(
    *,
    citation: str,
    image_index: object,
) -> str:
    """Build the label placed immediately before a retrieved image."""
    return RETRIEVED_IMAGE_EVIDENCE_PROMPT.format(
        citation=citation,
        image_index=image_index,
    )
