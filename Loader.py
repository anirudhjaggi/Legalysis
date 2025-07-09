"""
Example script showing how to use LangChain's TextLoader to load a .txt file
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import traceback

def load_text_file(file_path):
    """
    Load a text file using LangChain's TextLoader
    
    Args:
        file_path (str): Path to the .txt file
        
    Returns:
        list: List of Document objects (empty list if failed)
    """
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        if not documents:
            print("Warning: No documents loaded.")
            return []

        print(f"Successfully loaded {len(documents)} document(s)")
        print(f"Document metadata: {documents[0].metadata}")
        print(f"Document content preview (first 200 chars): {documents[0].page_content[:200]}...")

        return documents

    except Exception as e:
        print(f"Error loading file: {e}")
        traceback.print_exc()
        return []  # Never return None


def load_and_split_text(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a text file and split it into chunks
    
    Args:
        file_path (str): Path to the .txt file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of Document objects (empty list if failed)
    """
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        if not documents:
            print("Warning: No documents loaded for splitting.")
            return []

        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )

        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            print("Warning: No chunks were created.")
            return []

        print(f"Split document into {len(split_docs)} chunks")
        for i, doc in enumerate(split_docs[:3]):
            print(f"\nChunk {i+1} preview: {doc.page_content[:100]}...")

        return split_docs

    except Exception as e:
        print(f"Error processing file: {e}")
        traceback.print_exc()
        return []  # Never return None


# Example usage
if __name__ == "__main__":
    sample_file = "sample_text.txt"

    print("=== Basic Text Loading ===")
    documents = load_text_file(sample_file)

    print("\n=== Text Loading with Splitting ===")
    split_documents = load_and_split_text(sample_file, chunk_size=1000, chunk_overlap=100)

    print("\n=== Usage Instructions ===")
    print("To use this with your own .txt file:")
    print("1. Replace 'sample_text.txt' with your file path")
    print("2. Adjust chunk_size and chunk_overlap parameters as needed")
    print("3. Handle the returned Document objects for your specific use case")









# """
# Example script showing how to use LangChain's TextLoader to load a .txt file
# """

# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# import os

# def load_text_file(file_path):
#     """
#     Load a text file using LangChain's TextLoader
    
#     Args:
#         file_path (str): Path to the .txt file
        
#     Returns:
#         list: List of Document objects
#     """
#     try:
#         # Initialize the TextLoader
#         loader = TextLoader(file_path, encoding='utf-8')
        
#         # Load the document
#         documents = loader.load()
        
#         print(f"Successfully loaded {len(documents)} document(s)")
#         print(f"Document metadata: {documents[0].metadata}")
#         print(f"Document content preview (first 200 chars): {documents[0].page_content[:200]}...")
        
#         return documents
        
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return None

# def load_and_split_text(file_path, chunk_size=1000, chunk_overlap=200):
#     """
#     Load a text file and split it into chunks
    
#     Args:
#         file_path (str): Path to the .txt file
#         chunk_size (int): Size of each text chunk
#         chunk_overlap (int): Overlap between chunks
        
#     Returns:
#         list: List of Document objects (chunks)
#     """
#     try:
#         # Load the document
#         loader = TextLoader(file_path, encoding='utf-8')
#         documents = loader.load()
        
#         # Initialize text splitter
#         text_splitter = CharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separator="\n"
#         )
        
#         # Split the document into chunks
#         split_docs = text_splitter.split_documents(documents)
        
#         print(f"Split document into {len(split_docs)} chunks")
#         for i, doc in enumerate(split_docs[:3]):  # Show first 3 chunks
#             print(f"\nChunk {i+1} preview: {doc.page_content[:100]}...")
            
#         return split_docs
        
#     except Exception as e:
#         print(f"Error processing file: {e}")
#         return None








# # Example usage
# if __name__ == "__main__":
#     # Example 1: Basic text loading
#     print("=== Basic Text Loading ===")
    
#     # First, let's create a sample text file for demonstration
#     sample_file = "sample_text.txt"
    
#     # Load the text file
#     documents = load_text_file(sample_file)
    
#     print("\n=== Text Loading with Splitting ===")
    
#     # Load and split the text
#     split_documents = load_and_split_text(sample_file, chunk_size=1000, chunk_overlap=100)
    
#     # Clean up sample file
#     """if os.path.exists(sample_file):
#         os.remove(sample_file)
#         print(f"\nCleaned up sample file: {sample_file}")"""
    
#     print("\n=== Usage Instructions ===")
#     print("To use this with your own .txt file:")
#     print("1. Replace 'sample_document.txt' with your file path")
#     print("2. Adjust chunk_size and chunk_overlap parameters as needed")
#     print("3. Handle the returned Document objects for your specific use case")