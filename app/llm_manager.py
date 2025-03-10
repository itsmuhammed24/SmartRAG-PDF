# llm_manager.py
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

class LLMManager:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def get_response(self, query, matching_docs):
        """Génère une réponse en fonction des documents pertinents"""
        return self.chain.run(input_documents=matching_docs, question=query)
