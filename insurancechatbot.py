
import streamlit as st
import PyPDF2
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


def process_documents(upload_folder):
    text_data = ""
    for file in os.listdir(upload_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(upload_folder, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(pages)
            return texts
    return []


def initialize_chatbot(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.3),
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return chatbot


st.set_page_config(page_title="Insurance Policies Information Chatbot")
st.title(" Insurance Policy Information Chatbot")

st.sidebar.header("Upload Insurance Policy PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF", accept_multiple_files=True, type="pdf")

if uploaded_files:
    upload_folder = "uploaded_docs"
    os.makedirs(upload_folder, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(upload_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success("Files uploaded successfully!")

    with st.spinner("Processing documents and initializing chatbot..."):
        documents = process_documents(upload_folder)
        chatbot = initialize_chatbot(documents)

    st.success("Chatbot is ready to assist you!")

 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about insurance policies:", key="user_input")

    if user_input:
        result = chatbot({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_input, result["answer"]))

    for i, (user_q, bot_a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {bot_a}")

else:
    st.info("Please upload at least one PDF file to build the chatbot knowledge base.")
