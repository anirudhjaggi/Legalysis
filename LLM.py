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
            template="""You are a helpful AI assistant. Use the following context to answer the question. 
If the answer is not found in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
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














# """
# RAG Chatbot with Google Gemini LLM Integration

# This script integrates with your existing ChromaDB retriever and adds
# Google Gemini LLM functionality to create a complete RAG chatbot.
# """

# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from typing import List, Dict, Any
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class RAGChatbot:
#     """
#     RAG Chatbot that integrates retriever with Google Gemini LLM
#     """
    
#     def __init__(
#         self,
#         retriever,  # Your existing retriever instance
#         model_name: str = "gemini-pro",
#         temperature: float = 0.1,
#         max_tokens: int = 1000
#     ):
#         """
#         Initialize the RAG chatbot with existing retriever
        
#         Args:
#             retriever: Your existing ChromaDB retriever instance
#             model_name: Gemini model name
#             temperature: LLM temperature
#             max_tokens: Maximum tokens in response
#         """
#         # Load environment variables
#         load_dotenv()
        
#         # Get API key from environment
#         self.google_api_key = os.getenv("GOOGLE_API_KEY")
#         if not self.google_api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
#         # Store the retriever
#         self.retriever = retriever
        
#         # Initialize Google Gemini LLM
#         logger.info(f"Initializing Gemini model: {model_name}")
#         self.llm = ChatGoogleGenerativeAI(
#             model=model_name,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             google_api_key=self.google_api_key
#         )
        
#         # Create custom prompt template
#         self.prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
# If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

# Context:
# {context}

# Question: {question}

# Answer:"""
#         )
        
#         logger.info("RAG Chatbot initialized successfully")
    
#     def format_context(self, documents: List[Document]) -> str:
#         """
#         Format retrieved documents into context string
        
#         Args:
#             documents: List of retrieved documents
            
#         Returns:
#             Formatted context string
#         """
#         context_parts = []
#         for i, doc in enumerate(documents, 1):
#             context_parts.append(f"Document {i}:\n{doc.page_content}\n")
        
#         return "\n".join(context_parts)
    
#     def query(self, question: str, k: int = 4) -> Dict[str, Any]:
#         """
#         Query the RAG chatbot
        
#         Args:
#             question: User question
#             k: Number of documents to retrieve
            
#         Returns:
#             Dictionary with answer and metadata
#         """
#         logger.info(f"Processing query: {question}")
        
#         try:
#             # Retrieve relevant documents
#             retrieved_docs = self.retriever.search_documents(question, k=k)
            
#             # Format context from retrieved documents
#             context = self.format_context(retrieved_docs)
            
#             # Create the prompt
#             prompt = self.prompt_template.format(context=context, question=question)
            
#             # Get response from LLM
#             response = self.llm.invoke(prompt)
            
#             # Extract answer text
#             answer = response.content if hasattr(response, 'content') else str(response)
            
#             # Prepare response
#             result = {
#                 "question": question,
#                 "answer": answer,
#                 "num_sources": len(retrieved_docs),
#                 "sources": [
#                     {
#                         "content": doc.page_content,
#                         "metadata": doc.metadata
#                     }
#                     for doc in retrieved_docs
#                 ]
#             }
            
#             logger.info(f"Query processed successfully. Used {result['num_sources']} sources.")
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
#             return {
#                 "question": question,
#                 "answer": f"Sorry, I encountered an error: {str(e)}",
#                 "num_sources": 0,
#                 "sources": []
#             }
    
#     def chat(self):
#         """
#         Interactive chat interface
#         """
#         print("🤖 RAG Chatbot initialized!")
#         print("💬 Type 'quit' to exit, 'help' for commands\n")
        
#         while True:
#             try:
#                 user_input = input("You: ").strip()
                
#                 if user_input.lower() in ['quit', 'exit', 'q']:
#                     print("👋 Goodbye!")
#                     break
                
#                 elif user_input.lower() == 'help':
#                     self.show_help()
#                     continue
                
#                 elif not user_input:
#                     print("Please enter a question.")
#                     continue
                
#                 # Process the query
#                 response = self.query(user_input)
                
#                 # Display response
#                 print(f"\n🤖 Assistant: {response['answer']}")
#                 print(f"📄 Sources used: {response['num_sources']}")
                
#                 # Optionally show sources
#                 if response['num_sources'] > 0:
#                     show_sources = input("\nShow sources? (y/n): ").lower() == 'y'
#                     if show_sources:
#                         self.display_sources(response['sources'])
                
#                 print("\n" + "-" * 50 + "\n")
                
#             except KeyboardInterrupt:
#                 print("\n👋 Goodbye!")
#                 break
#             except Exception as e:
#                 print(f"❌ Error: {e}")
#                 logger.error(f"Error in chat: {e}")
    
#     def show_help(self):
#         """Show help information"""
#         print("\n📋 Available commands:")
#         print("  help  - Show this help message")
#         print("  quit  - Exit the chatbot")
#         print("  Just type your question to get an answer!\n")
    
#     def display_sources(self, sources: List[Dict[str, Any]]):
#         """Display source documents"""
#         print("\n📚 Source Documents:")
#         for i, source in enumerate(sources, 1):
#             print(f"\n--- Source {i} ---")
#             print(f"Content: {source['content'][:300]}...")
#             if source['metadata']:
#                 print(f"Metadata: {source['metadata']}")


# def main():
#     """
#     Example usage with your existing retriever
#     """
    
#     # Import your existing retriever here
#     # from your_retriever_module import YourRetrieverClass
    
#     print("🚀 Starting RAG Chatbot Application")
#     print("=" * 50)
    
#     # Check for .env file
#     if not os.path.exists('.env'):
#         print("❌ Error: .env file not found!")
#         print("Please create a .env file with your GOOGLE_API_KEY.")
#         print("You can copy .env.example to .env and update it.")
#         return
    
#     try:
#         # Initialize your existing retriever
#         # retriever = YourRetrieverClass(
#         #     persist_directory="path/to/chromadb",
#         #     collection_name="langchain"
#         # )
        
#         # For demonstration, assuming you have a retriever instance
#         # Replace this with your actual retriever initialization
#         print("⚠️  Please uncomment and update the retriever initialization code")
#         print("Replace the placeholder with your actual retriever instance")
#         return
        
#         # Initialize chatbot with your retriever
#         chatbot = RAGChatbot(retriever=retriever)
        
#         # Start interactive chat
#         chatbot.chat()
        
#     except Exception as e:
#         print(f"❌ Error initializing chatbot: {e}")
#         logger.error(f"Initialization error: {e}")


# def example_programmatic_usage():
#     """
#     Example of programmatic usage
#     """
    
#     # Initialize your retriever
#     # retriever = YourRetrieverClass(...)
    
#     # Initialize chatbot
#     # chatbot = RAGChatbot(retriever=retriever)
    
#     # Example queries
#     questions = [
#         "What is artificial intelligence?",
#         "How does machine learning work?",
#         "What are the benefits of deep learning?"
#     ]
    
#     # for question in questions:
#     #     print(f"\nQuestion: {question}")
#     #     response = chatbot.query(question, k=3)
#     #     print(f"Answer: {response['answer']}")
#     #     print(f"Sources: {response['num_sources']}")
#     #     print("-" * 50)
    
#     pass


# # Simple integration example
# class SimpleRAGChat:
#     """
#     Simplified version for quick integration
#     """
    
#     def __init__(self, retriever):
#         load_dotenv()
#         self.retriever = retriever
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             temperature=0.1,
#             google_api_key=os.getenv("GOOGLE_API_KEY")
#         )
    
#     def ask(self, question: str, k: int = 4) -> str:
#         """
#         Simple question-answer interface
        
#         Args:
#             question: User question
#             k: Number of documents to retrieve
            
#         Returns:
#             Answer string
#         """
#         # Retrieve documents
#         docs = self.retriever.search_documents(question, k=k)
        
#         # Format context
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # Create prompt
#         prompt = f"""Based on the following context, answer the question:

# Context:
# {context}

# Question: {question}

# Answer:"""
        
#         # Get response
#         response = self.llm.invoke(prompt)
#         return response.content if hasattr(response, 'content') else str(response)


# if __name__ == "__main__":
#     main()