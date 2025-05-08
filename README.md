Gemini-Powered RAG Q&A Assistant


Features

Upload a PDF or TXT file

Paste a URL to scrape article content

Ask questions about the uploaded or linked content

Uses Google's Gemini API to generate context-aware answers

Embeds content with HuggingFace models + FAISS or ChromaDB

Interactive web interface via Streamlit

Setup Instructions:

Clone the repository

git clone  https://github.com/Dharmaraj-tech/Gemini-Powered-RAG-Question-Answering-App-with-Streamlit.git

 Install dependencies
 pip install streamlit langchain langchain-community google-generativeai newspaper3k sentence-transformers lxml_html_clean

Generate your api key in  https://makersuite.google.com/app/apikey
replace it in thhe code Gemini API key

Run the Script
streamlit run Gemini-Powered RAG Question Answering App with Streamlit.py
