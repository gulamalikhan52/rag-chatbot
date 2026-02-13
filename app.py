import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline


# ---------------- PAGE ----------------
st.set_page_config(
    page_title="📚 Project Chatbot",
    page_icon="🤖"
)

st.title("📚 Project Chatbot (RAG)")
st.write("Ask questions from your PDF")


# ---------------- LOAD PDF ----------------
@st.cache_resource
def load_docs():

    loader = PyPDFLoader("data/how_hackers_hack_systems_full_pages.pdf")   # 👈 change name
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    docs = splitter.split_documents(pages)

    return docs


# ---------------- BUILD RAG ----------------
@st.cache_resource
def build_rag():

    docs = load_docs()

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embedding_model)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    # LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",   # better for cloud
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer using ONLY the context.
        If not found, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    # RAG Chain
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = build_rag()


# ---------------- CHAT UI ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Input
question = st.chat_input("Ask from the PDF...")


if question:

    # Show user msg
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Bot reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            answer = rag_chain.invoke(question)

            st.markdown(answer)

    # Save bot msg
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
