# vector_db.py
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class VectorDBManager:
    def __init__(self, persist_directory="chroma_store"):
        self.persist_directory = persist_directory
        self.client = chromadb.Client()
        self.collection_name = "consent_collection"

    def create_or_load_db(self, chunked_documents):
        """Crée ou charge la base de données Chroma"""
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            persist_directory=self.persist_directory
        )

        if chunked_documents:
            if not self.client.list_collections():
                self.client.create_collection(self.collection_name)
            vectordb.add_documents(chunked_documents)
            vectordb.persist()

        return vectordb

    def search(self, vectordb, query, top_k=5):
        """Effectue une recherche par similarité"""
        return vectordb.similarity_search(query, k=top_k)
