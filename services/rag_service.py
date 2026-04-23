"""
services/rag_service.py
=======================
RAG pipeline using ChromaDB with local sentence-transformers embeddings.
No API key required — model downloads once (~22MB) and runs fully offline.
"""

import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from core.config import get_settings


class RAGService:
    """
    ChromaDB knowledge base with local sentence-transformers embeddings.
    Auto-loads persisted DB on startup; builds from markdown on first run.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = self._load_or_build()

    def _load_or_build(self) -> Chroma:
        persist_dir = self._settings.chroma_persist_dir
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self._embeddings,
            )
        return self._build()

    def _build(self) -> Chroma:
        print("  📚  Building knowledge base (first run — takes ~10s)...")
        loader = TextLoader(self._settings.kb_path, encoding="utf-8")
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(raw_docs)

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=self._settings.chroma_persist_dir,
        )
        print("  ✅  Knowledge base ready.")
        return vs

    def retrieve(self, query: str) -> str:
        docs = self._vectorstore.similarity_search(
            query, k=self._settings.retrieval_k
        )
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def rebuild(self) -> None:
        """Force full rebuild — delete old DB, re-embed from markdown."""
        # Close the current vectorstore connection before deleting the directory
        try:
            self._vectorstore._client.reset()
        except Exception:
            pass
        self._vectorstore = None

        persist_dir = self._settings.chroma_persist_dir
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        self._vectorstore = self._build()
