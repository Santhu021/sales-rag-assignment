import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

#1. Load_pdf

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text+=content

    return text

#2. Split text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    return splitter.split_text(text)

#3. Initialize Pinecone

pc = Pinecone(api_key = os.getenv('PINECONE_API_KEY'))

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

#4. Store in Pinecone

def store_in_pinecone(chunks):
    vectors = []

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        vectors.append({
            "id":str(i),
            "values":vector,
            "metadata":{"text":chunk}
        })

    index.upsert(vectors)
    print("Data stored in Pineconse sucessfully")

#Main Pipeline

if __name__ == "__main__":
    file_path = r"C:\Users\lenovo\Desktop\NGO\sample.pdf"

    print("loading pdf")
    text = load_pdf(file_path)

    print("splitting text")
    chunks = split_text(text)

    print(f"chunks created:{len(chunks)}")

    print("Storing in Pinecone")
    store_in_pinecone(chunks)