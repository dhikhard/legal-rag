import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a legal assistant specializing in Indian law.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this in the provided documents."
Always mention the relevant section number when answering.
Be clear, concise and helpful.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    from retriever import retrieve
    query = "What is the punishment for murder?"
    chunks, metadatas = retrieve(query)
    answer = generate_answer(query, chunks)
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for meta in metadatas:
        print(f"  {meta['source']} | chunk {meta['chunk_index']}")
