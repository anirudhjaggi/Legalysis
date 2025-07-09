"""
ChromaDB Retriever with Google Gemini API

This script creates a retriever that connects to pre-existing embeddings
stored in ChromaDB and uses Google Gemini API for LLM functionality.
"""

import numpy as np
from Embedder import LegalEmbedder
import chromadb
import os
import traceback

class Retriever:
    """
    Retriever that uses LegalEmbedder for query embedding and ChromaDB for retrieval.
    """
    def __init__(self, persist_directory: str, collection_name: str = "legal_documents", model_name: str = "law-ai/InLegalBERT"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = LegalEmbedder(model_name=model_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(self, query: str, k: int = 4):
        """
        Retrieve top-k relevant documents for the query.
        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.
        Returns:
            List of dicts with 'document', 'id', and 'score'.
        """
        try:
            # Embed the query
            query_embedding = self.embedder.embed_texts([query])[0]

            # Perform query in ChromaDB (NOTE: 'ids' removed from 'include')
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "distances"]
            )

            # Format results
            docs = []
            for i in range(len(results["documents"][0])):
                docs.append({
                    "document": results["documents"][0][i],
                    "id": results["ids"][0][i],  # 'ids' are always included by default
                    "score": results["distances"][0][i]
                })
            return docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            traceback.print_exc()
            return []

def main():
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "sample_documents"
    MODEL_NAME = "law-ai/InLegalBERT"

    print("=== Legal Document Retriever ===")
    print(f"ChromaDB directory: {PERSIST_DIRECTORY}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Embedding model: {MODEL_NAME}")

    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Error: ChromaDB directory '{PERSIST_DIRECTORY}' does not exist.")
        return

    try:
        retriever = Retriever(
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )
        print("Retriever initialized successfully!\n")

        # Example query
        query = "What is contract law?"
        print(f"Query: {query}")
        results = retriever.retrieve(query, k=3)

        print(f"\nTop {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"ID: {doc['id']}")
            print(f"Score (distance): {doc['score']}")
            print(f"Content: {doc['document'][:300]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure ChromaDB directory and collection exist and contain embeddings.")
        print("2. The embedding model must match the one used for storage.")
        print("3. Check that Embedder.py and chromadb are installed and working.")

if __name__ == "__main__":
    main()














# """
# ChromaDB Retriever with Google Gemini API

# This script creates a retriever that connects to pre-existing embeddings
# stored in ChromaDB and uses Google Gemini API for LLM functionality.
# """

# import numpy as np
# from Embedder import LegalEmbedder
# import chromadb
# import os

# class Retriever:
#     """
#     Retriever that uses LegalEmbedder for query embedding and ChromaDB for retrieval.
#     """
#     def __init__(self, persist_directory: str, collection_name: str = "legal_documents", model_name: str = "law-ai/InLegalBERT"):
#         self.persist_directory = persist_directory
#         self.collection_name = collection_name
#         self.embedder = LegalEmbedder(model_name=model_name)
#         self.client = chromadb.PersistentClient(path=persist_directory)
#         self.collection = self.client.get_collection(name=collection_name)

#     def retrieve(self, query: str, k: int = 4):
#         """
#         Retrieve top-k relevant documents for the query.
#         Args:
#             query (str): The search query.
#             k (int): Number of documents to retrieve.
#         Returns:
#             List of dicts with 'document', 'id', and 'score'.
#         """
#         # Embed the query using LegalEmbedder
#         query_embedding = self.embedder.embed_texts([query])[0]  # shape: (embedding_dim,)
#         # ChromaDB expects a list of embeddings
#         results = self.collection.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=k,
#             include=["documents", "distances", "ids"]
#         )
#         # Format results
#         docs = []
#         for i in range(len(results["documents"][0])):
#             docs.append({
#                 "document": results["documents"][0][i],
#                 "id": results["ids"][0][i],
#                 "score": results["distances"][0][i]
#             })
#         return docs

# def main():
#     # Update these paths as needed
#     PERSIST_DIRECTORY = "./chroma_db"  # Path to your ChromaDB directory
#     COLLECTION_NAME = "sample_documents"  # Name of your ChromaDB collection
#     MODEL_NAME = "law-ai/InLegalBERT"  # Model used for embedding (should match the one used for storage)

#     print("=== Legal Document Retriever ===")
#     print(f"ChromaDB directory: {PERSIST_DIRECTORY}")
#     print(f"Collection name: {COLLECTION_NAME}")
#     print(f"Embedding model: {MODEL_NAME}")

#     if not os.path.exists(PERSIST_DIRECTORY):
#         print(f"Error: ChromaDB directory '{PERSIST_DIRECTORY}' does not exist.")
#         return

#     try:
#         retriever = Retriever(
#             persist_directory=PERSIST_DIRECTORY,
#             collection_name=COLLECTION_NAME,
#             model_name=MODEL_NAME
#         )
#         print("Retriever initialized successfully!\n")
#         # Example query
#         query = "What is contract law?"
#         print(f"Query: {query}")
#         results = retriever.retrieve(query, k=3)
#         print(f"\nTop {len(results)} results:")
#         for i, doc in enumerate(results, 1):
#             print(f"\n--- Result {i} ---")
#             print(f"ID: {doc['id']}")
#             print(f"Score (distance): {doc['score']}")
#             print(f"Content: {doc['document'][:300]}...")
#     except Exception as e:
#         print(f"Error: {e}")
#         print("\nTroubleshooting:")
#         print("1. Ensure ChromaDB directory and collection exist and contain embeddings.")
#         print("2. The embedding model must match the one used for storage.")
#         print("3. Check that Embedder.py and chromadb are installed and working.")

# if __name__ == "__main__":
#     main()