from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever
from langchain.vectorstores.base import VectorStore


class ProcessQuery():
    load_dotenv()
    def __init__(self):
        pass

    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        print("I am here 3 \n", type(vectorstore))
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        return conversation_chain