import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# -----------------------
# OpenAI LLM
# -----------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------
# Embeddings
# -----------------------
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------
# Pinecone setup
# -----------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("sales-rag")


# -----------------------
# RAG FUNCTION
# -----------------------
def get_answer_from_docs(user_question):

    # Convert question into embedding vector
    query_vector = embeddings.embed_query(user_question)

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # Extract retrieved chunks
    context = "\n\n".join(
        match["metadata"]["text"]
        for match in results["matches"]
    )

    # Prompt
    prompt = f"""
You are a helpful AI assistant.

RULES:
- Use ONLY the provided context
- If answer is not available, say "I don't know"
- Do not hallucinate

Context:
{context}

Question:
{user_question}
"""

    # Generate response
    response = llm.invoke(prompt)

    return response.content


# -----------------------
# TEST LOOP
# -----------------------
if __name__ == "__main__":

    while True:
        question = input("\nAsk a question: ")

        answer = get_answer_from_docs(question)

        print("\nAnswer:")
        print(answer)