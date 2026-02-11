

# ======================== FILE: processors.py ========================
"""
File processors — each file type gets parsed, chunked, and prepared for indexing.
"""
import base64, pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
from llm import llm_vision
import re

# ----------- TEXT / PDF -----------
class PDFProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, file_path: str, metadata: dict) -> dict:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for p in pages:
            p.metadata.update(metadata)

        # --- Standard chunks (for Naive RAG, HyDE RAG, etc.) ---
        standard_chunks = self.splitter.split_documents(pages)

        # --- Sentence-level chunks (for Sentence Window RAG) ---
        sentence_chunks = []
        for page in pages:
            sentences = re.split(r'(?<=[.!?])\s+', page.page_content)
            for i, sent in enumerate(sentences):
                if sent.strip():
                    # Store surrounding sentences as metadata
                    window_start = max(0, i - settings.sentence_window_size)
                    window_end = min(len(sentences), i + settings.sentence_window_size + 1)
                    window_text = " ".join(sentences[window_start:window_end])

                    sentence_chunks.append(Document(
                        page_content=sent.strip(),
                        metadata={
                            **metadata,
                            "window_text": window_text,
                            "sentence_index": i,
                            "page": page.metadata.get("page", 0),
                        }
                    ))

        # --- Parent-child chunks (for Parent-Child RAG) ---
        # Parents = large chunks, Children = small sub-chunks
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
        parent_chunks = parent_splitter.split_documents(pages)
        child_chunks = []
        for i, parent in enumerate(parent_chunks):
            parent.metadata["parent_id"] = f"parent_{metadata.get('file_id', '')}_{i}"
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent.metadata["parent_id"]
                child.metadata["parent_content"] = parent.page_content
            child_chunks.extend(children)

        # --- Full text for Graph RAG entity extraction ---
        full_text = "\n".join([p.page_content for p in pages])

        return {
            "standard_chunks": standard_chunks,
            "sentence_chunks": sentence_chunks,
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks,
            "full_text": full_text,
            "file_type": "pdf",
        }


# ----------- IMAGE -----------
class ImageProcessor:
    def process(self, file_path: str, metadata: dict) -> dict:
        with open(file_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        description = llm_vision(
            img_b64,
            "Describe this image in comprehensive detail for search indexing. "
            "Include: objects, text/labels, colors, layout, data shown, relationships between elements."
        )

        doc = Document(
            page_content=description,
            metadata={**metadata, "image_path": file_path, "modality": "image",
                      "image_b64_preview": img_b64[:500]},  # store preview for retrieval
        )
        return {
            "standard_chunks": [doc],
            "sentence_chunks": [doc],
            "child_chunks": [doc],
            "parent_chunks": [doc],
            "full_text": description,
            "file_type": "image",
        }


# ----------- CSV / EXCEL -----------
class CSVProcessor:
    def process(self, file_path: str, metadata: dict) -> dict:
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

        # Schema chunk — always retrieved for table queries
        schema = f"TABLE: {metadata.get('source_filename', 'data')}\n"
        schema += f"COLUMNS: {', '.join(df.columns)}\n"
        schema += f"ROWS: {len(df)} | DTYPES: {dict(df.dtypes.astype(str))}\n"
        schema += f"SAMPLE:\n{df.head(5).to_string()}\n"
        schema += f"STATS:\n{df.describe().to_string()}"

        schema_doc = Document(
            page_content=schema,
            metadata={**metadata, "chunk_type": "schema", "columns": list(df.columns),
                      "row_count": len(df)},
        )

        # Row group chunks
        row_chunks = []
        for i in range(0, len(df), 25):
            group = df.iloc[i:i+25]
            text = f"Rows {i}-{i+len(group)} of {metadata.get('source_filename', 'data')}:\n{group.to_string()}"
            row_chunks.append(Document(
                page_content=text,
                metadata={**metadata, "chunk_type": "rows", "row_start": i},
            ))

        # Store raw CSV path for text-to-pandas agent
        schema_doc.metadata["csv_path"] = file_path

        all_chunks = [schema_doc] + row_chunks
        return {
            "standard_chunks": all_chunks,
            "sentence_chunks": all_chunks,
            "child_chunks": all_chunks,
            "parent_chunks": [schema_doc],
            "full_text": schema,
            "file_type": "csv",
            "csv_path": file_path,
        }
