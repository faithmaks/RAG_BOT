import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from bs4 import BeautifulSoup
import requests
from openai import OpenAI, RateLimitError, APIError
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Initialize Chroma
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag-chat", embedding_function=embedding_function)

# --- Helper Functions ---

def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    return "\n\n".join(texts)

def scrape_website(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator="\n")
        return text
    except Exception as e:
        return f"Error scraping site: {str(e)}"

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def add_to_chroma(text_chunks):
    ids = [str(i) for i in range(len(text_chunks))]
    collection.add(documents=text_chunks, ids=ids)

# --- NEW: Augment query with GPT ---
def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
Provide an example answer to the given question, that might be found in a document like an annual report."""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Failed to augment query: {e}")
        return ""

# --- Final RAG answer logic ---
def answer_question(question):
    try:
        hypothetical_answer = augment_query_generated(question)
        joint_query = f"{question} {hypothetical_answer}" if hypothetical_answer else question

        results = collection.query(
            query_texts=[joint_query],
            n_results=5,
            include=["documents"]
        )
        retrieved_docs = results["documents"][0]
        context = "\n\n".join(retrieved_docs)

        prompt = f"""You are a helpful assistant that answers questions using the following document context.

Context:
{context}

Question: {question}
Answer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You answer questions based on context provided."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        return f"⚠️ OpenAI API rate limit or quota exceeded."
    except APIError as e:
        return f"⚠️ OpenAI API error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# --- Streamlit UI ---

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📊 Document + Website Q&A Chatbot")

with st.sidebar:
    st.header("Upload or Crawl")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    website_url = st.text_input("Or enter a website URL")

    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            pdf_text = process_pdf(uploaded_pdf)
            chunks = split_text(pdf_text)
            add_to_chroma(chunks)
            st.success("✅ PDF processed and indexed.")

    if website_url:
        with st.spinner("Scraping website..."):
            site_text = scrape_website(website_url)
            if "Error" not in site_text:
                chunks = split_text(site_text)
                add_to_chroma(chunks)
                st.success("✅ Website scraped and indexed.")
            else:
                st.error(site_text)

st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)
        st.markdown("### ✅ Answer")
        st.markdown(answer)

        feedback = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"])
        if feedback:
            st.success("Thanks for your feedback!")
