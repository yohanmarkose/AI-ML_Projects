from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings


class ProcessDocs():
    load_dotenv()  # Loading the env variables ( API keys/tokens)
    def __init__(self):
        pass

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_content = PdfReader(pdf)
            for page in pdf_content.pages:
                text += page.extract_text()
        return text
    

    def get_text_chunks(self, raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks
    
    def create_vector_store(self, text_chunks):
        # embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        return vectorstore
    
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