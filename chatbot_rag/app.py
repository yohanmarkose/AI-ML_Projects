import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient

from process_docs import ProcessDocs
from htmlTemplates import css, bot_template, user_template

load_dotenv()

def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_content = PdfReader(pdf)
            for page in pdf_content.pages:
                text += page.extract_text()
        return text
    

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def get_conversation_chain(vectorstore, model_choice):
    if model_choice == "OpenAI(Paid)":
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def handle_user_input(user_input):
    response = st.session_state.conversation({"question": user_input})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:  # to check If it is user input
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    


def main():
    # load_dotenv()  # Loading the env variables ( API keys/tokens)
    # docs = ProcessDocs()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Chat with multiple Docs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple Docs :books:")

    # temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)  # Slider

    user_input = st.text_input("What do you want to know about your documents?", placeholder="Type in your queries here..")

    if user_input:
        handle_user_input(user_input)

        # chat_model = ChatOpenAI()
        # message = HumanMessage(content=str(user_input))
        # response = chat_model.predict_messages([message])
        # st.write(response.content)

    with st.sidebar:
        st.subheader("Model")
        model_choice = st.selectbox("Select your model", ["OpenAI(Paid)", "HuggingFaceHub(Free)"])

        st.subheader("Yours Docs")
        files = st.file_uploader("Upload your Docs", accept_multiple_files=True)
        if st.button("Process Docs"):
            with st.spinner("Processing"):
                # Get the PDF text
                raw_pdf_text = get_pdf_text(files)
                # Get the Text chunks
                text_chunks = get_text_chunks(raw_pdf_text)
                # Create the Vector Store
                vector_store = create_vector_store(text_chunks)
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store, model_choice)

if __name__ == "__main__":
    main()