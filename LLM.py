"""
RAG Chatbot with Google Gemini LLM Integration
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    RAG Chatbot integrating Retriever and Google Gemini LLM
    """
    def __init__(
        self,
        retriever,  # Must have a .retrieve() method
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.5,
        max_tokens: int = 10000
    ):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment. Check .env file.")

        self.retriever = retriever

        logger.info(f"Initializing Gemini model: {model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=self.google_api_key
        )

        self.prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable AI legal assistant trained on Indian law, including the Indian Penal Code (IPC).
Use the provided context as your **primary source**, but you are allowed to fill in **general legal knowledge** only when the context is insufficient.

**Guidelines:**
- Always **prioritize** and **cite** from the provided context when possible.
- You may include basic legal reasoning or general Indian legal principles **only if** the context does not directly address the question.
- Clearly indicate when you're using general knowledge versus direct references from context.
- Avoid speculation, personal opinions, or legal advice.

Now, using the above instructions, answer the user's question in a **structured and informative format**.

Context:
{context}

User's Question: {question}

Provide your answer in this format:
1. **Summary**: A one-line conclusion or short answer
2. **Relevant IPC Sections**: Bullet points with section numbers and short descriptions (cite from context if possible)
3. **Explanation**: A short paragraph explaining the reasoning behind the answer
4. **Caveats / Additional Notes**: Any limitations, assumptions, or suggestions to consult a lawyer

Answer:
"""
)


        logger.info("RAG Chatbot initialized.")

    def format_context(self, documents: List[Document]) -> str:
        return "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)
        )

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        logger.info(f"Processing query: {question}")
        try:
            raw_results = self.retriever.retrieve(question, k=k)

            retrieved_docs = [
                Document(
                    page_content=doc["document"],
                    metadata={"id": doc["id"], "score": doc["score"]}
                )
                for doc in raw_results
            ]

            context = self.format_context(retrieved_docs)
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)

            return {
                "question": question,
                "answer": response.content if hasattr(response, 'content') else str(response),
                "num_sources": len(retrieved_docs),
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "question": question,
                "answer": f"Error: {e}",
                "num_sources": 0,
                "sources": []
            }

    def chat(self):
        print("🤖 RAG Chatbot ready. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting chatbot.")
                break

            if not user_input:
                print("⚠️ Please enter a valid question.")
                continue

            response = self.query(user_input)
            print(f"\n🤖 Answer: {response['answer']}")
            print(f"📚 Sources used: {response['num_sources']}")

            if response['num_sources'] > 0:
                if input("Show sources? (y/n): ").lower() == "y":
                    for i, src in enumerate(response['sources'], 1):
                        print(f"\n--- Source {i} ---")
                        print(f"Content: {src['content'][:300]}...")
                        print(f"Metadata: {src['metadata']}")

            print("\n" + "-" * 50 + "\n")


# For standalone use (optional)
def main():
    from Retriever import Retriever  # ✅ Update path if needed

    print("🚀 Initializing retriever...")
    retriever = Retriever(
        persist_directory="./chroma_db",
        collection_name="legal_documents",
        model_name="law-ai/InLegalBERT"
    )

    chatbot = RAGChatbot(retriever=retriever)
    chatbot.chat()


if __name__ == "__main__":
    main()