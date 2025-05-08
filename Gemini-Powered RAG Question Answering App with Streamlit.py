import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import google.generativeai as genai
from newspaper import Article
os.environ["GOOGLE_API_KEY"] = "AIzaSyBnY07rAPJOuuMsEB9nCA91VxgHzBuXT3g"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
def load_document(source, is_url=False):
    if is_url:
        article = Article(source)
        article.download()
        article.parse()
        return article.text
    elif source.endswith('.pdf'):
        loader = PyPDFLoader(source)
        return loader.load()[0].page_content
    else:
        loader = TextLoader(source)
        return loader.load()[0].page_content
def prepare_vector_store(text, use_chroma=False):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings) if use_chroma else FAISS.from_documents(docs, embeddings)
def ask_gemini(question, context):
    prompt = f"""Answer the question based only on the context below.\n\nContext: {context}\n\nQuestion: {question}"""
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
def main():
    st.title("RAG Q&A Assistant using Gemini")

    input_type = st.radio("Select Input Type", ("File Upload", "Web URL"))

    text = ""
    if input_type == "File Upload":
        uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            path = f"temp_{uploaded_file.name}"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            text = load_document(path)
    else:
        url = st.text_input("Enter the URL")
        if url:
            text = load_document(url, is_url=True)

    if text:
        db = prepare_vector_store(text)
        retriever = db.as_retriever()
        query = st.text_input("Ask a question about the content")
        if query:
            relevant_docs = retriever.invoke(query)
            context = " ".join([doc.page_content for doc in relevant_docs])
            answer = ask_gemini(query, context)
            st.markdown("### 🔹 Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()