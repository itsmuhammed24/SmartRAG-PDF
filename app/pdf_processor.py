# pdf_processor.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

class PDFProcessor:
    def __init__(self, pdf_folder_path: str):
        self.pdf_folder_path = pdf_folder_path

    def load_and_split_pdfs(self, chunk_size=1000, chunk_overlap=10):
        """Charge les fichiers PDF depuis le dossier et les divise en chunks"""
        documents = []
        
        for file in os.listdir(self.pdf_folder_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)
