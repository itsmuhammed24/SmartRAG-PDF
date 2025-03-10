# main.py
import os
from dotenv import load_dotenv
import streamlit as st
from pdf_processor import PDFProcessor
from vector_db import VectorDBManager
from llm_manager import LLMManager

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Query PDF Source")

# Définition du dossier contenant les PDF
PDF_FOLDER = "data/"

# Initialisation des composants
pdf_processor = PDFProcessor(PDF_FOLDER)
vector_db_manager = VectorDBManager()
llm_manager = LLMManager()

# Chargement et indexation des documents
chunked_documents = pdf_processor.load_and_split_pdfs()
vectordb = vector_db_manager.create_or_load_db(chunked_documents)

# Interface utilisateur
form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    matching_docs = vector_db_manager.search(vectordb, form_input)
    response = llm_manager.get_response(form_input, matching_docs)
    st.write(response)
