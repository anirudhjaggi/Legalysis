import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from Loader import load_and_split_text
import torch
import chromadb
import traceback
from tqdm import tqdm

class LegalEmbedder:
    def __init__(self, model_name: str = "law-ai/InLegalBERT"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("Model loaded successfully!")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Input to embed_texts must be a non-empty list of non-empty strings.")
        
        print(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings_list = []

        for i, text in enumerate(tqdm(texts, desc="🔄 Generating Embeddings", unit="chunk")):
            try:
                encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    output = self.model(**encoded_input)
                    last_hidden_state = output.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1)
                embeddings_list.append(embedding.squeeze().numpy())
            except Exception as e:
                print(f"Error processing text chunk {i}: {e}")
                traceback.print_exc()
                continue

        if not embeddings_list:
            raise RuntimeError("No valid embeddings were generated.")

        embeddings_array = np.array(embeddings_list)
        print(f"✅ Generated embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        if not output_path.endswith('.npy'):
            output_path += '.npy'
        np.save(output_path, embeddings)
        print(f"💾 Embeddings saved to: {output_path}")
        return output_path

    def store_embeddings_in_chroma(self, npy_file_path: str, text_chunks: List[str], 
                                   collection_name: str = "legal_documents", 
                                   batch_size: int = 5000, 
                                   persist_directory: str = "./chroma_db") -> bool:
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            print(f"🧠 ChromaDB client initialized at: {persist_directory}")

            embeddings = np.load(npy_file_path)
            print(f"📥 Loaded embeddings from: {npy_file_path}, shape: {embeddings.shape}")

            if len(embeddings) != len(text_chunks):
                print(f"⚠️ Mismatch: {len(embeddings)} embeddings vs {len(text_chunks)} text chunks")
                return False

            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Legal document embeddings"}
            )

            total_batches = (len(embeddings) + batch_size - 1) // batch_size
            print(f"📊 Total embeddings: {len(embeddings)} | Batch size: {batch_size}")

            for batch_idx in tqdm(range(total_batches), desc="📦 Storing to ChromaDB", unit="batch"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(embeddings))

                batch_embeddings = embeddings[start_idx:end_idx]
                batch_texts = text_chunks[start_idx:end_idx]
                batch_ids = [f"doc_{start_idx + i}" for i in range(len(batch_embeddings))]
                collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    ids=batch_ids
                )

            print("✅ All embeddings stored successfully.")
            return True

        except Exception as e:
            print(f"❌ Error in ChromaDB storage: {e}")
            traceback.print_exc()
            return False


# === Example Usage ===
if __name__ == "__main__":
    print("=== Loading and Splitting Text ===")
    split_docs = load_and_split_text("sample_text.txt", chunk_size=1000, chunk_overlap=200)

    if not split_docs:
        print("⚠️ WARNING: load_and_split_text returned empty list. Using fallback text.")
        sample_texts = [
            "This is the first legal document chunk about contract law.",
            "The second chunk discusses tort liability and damages.",
            "This third chunk covers constitutional law principles."
        ]
        text_chunks = sample_texts
    else:
        text_chunks = [doc.page_content for doc in split_docs if doc and doc.page_content.strip()]

    if not text_chunks:
        raise ValueError("❌ No valid text chunks to embed.")

    embedder = LegalEmbedder()
    embeddings = embedder.embed_texts(text_chunks)
    output_file = embedder.save_embeddings(embeddings, "sample_embeddings.npy")

    success = embedder.store_embeddings_in_chroma(
        npy_file_path=output_file,
        text_chunks=text_chunks,
        collection_name="sample_documents",
        batch_size=1000
    )

    if success:
        print("=== ChromaDB Embedding Preview ===")
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(name="sample_documents")

            results = collection.get(include=["embeddings", "documents"])

            if results \
            and results.get("documents") and len(results.get("documents")) > 0 \
            and results.get("embeddings") is not None and len(results.get("embeddings")) > 0:
    
                total = len(results['documents'])   
                print(f"📚 Total documents stored in ChromaDB: {total}")
                print(f"🔢 Embedding dimension: {len(results['embeddings'][0])}")

                for i in range(min(5, total)):
                    print(f"\n📄 Doc {i+1} ID: {results['ids'][i]}")
                    print(f"Text: {results['documents'][i][:300]}...")
                    print(f"Embedding Preview: {results['embeddings'][i][:10]}...")
            else:
                print("⚠️ No documents or embeddings found in ChromaDB collection.")

        except Exception as e:
            print(f"❌ Failed to preview ChromaDB contents: {e}")
            traceback.print_exc()

    else:
        print("❌ Failed to store embeddings in ChromaDB.")

    try:
        loaded_embeddings = np.load(output_file)
        print(f"✅ Verification: Loaded embeddings shape: {loaded_embeddings.shape}")
    except Exception as e:
        print(f"❌ Error loading saved embeddings: {e}")
        traceback.print_exc()














# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from typing import List
# from Loader import load_and_split_text
# import torch
# import chromadb
# import traceback

# class LegalEmbedder:
#     def __init__(self, model_name: str = "law-ai/InLegalBERT"):
#         print(f"Loading model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         print("Model loaded successfully!")

#     def embed_texts(self, texts: List[str]) -> np.ndarray:
#         if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
#             raise ValueError("Input to embed_texts must be a non-empty list of non-empty strings.")
        
#         print(f"Generating embeddings for {len(texts)} text chunks...")
#         embeddings_list = []

#         for i, text in enumerate(texts):
#             try:
#                 encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#                 with torch.no_grad():
#                     output = self.model(**encoded_input)
#                     last_hidden_state = output.last_hidden_state
#                 embedding = torch.mean(last_hidden_state, dim=1)
#                 embeddings_list.append(embedding.squeeze().numpy())
#             except Exception as e:
#                 print(f"Error processing text chunk {i}: {e}")
#                 traceback.print_exc()
#                 continue

#         if not embeddings_list:
#             raise RuntimeError("No valid embeddings were generated.")

#         embeddings_array = np.array(embeddings_list)
#         print(f"Generated embeddings shape: {embeddings_array.shape}")
#         return embeddings_array

#     def save_embeddings(self, embeddings: np.ndarray, output_path: str):
#         if not output_path.endswith('.npy'):
#             output_path += '.npy'
#         np.save(output_path, embeddings)
#         print(f"Embeddings saved to: {output_path}")
#         return output_path

#     def store_embeddings_in_chroma(self, npy_file_path: str, text_chunks: List[str], 
#                                    collection_name: str = "legal_documents", 
#                                    batch_size: int = 5000, 
#                                    persist_directory: str = "./chroma_db") -> bool:
#         try:
#             client = chromadb.PersistentClient(path=persist_directory)
#             print(f"ChromaDB client initialized at: {persist_directory}")

#             embeddings = np.load(npy_file_path)
#             print(f"Loaded embeddings from: {npy_file_path}, shape: {embeddings.shape}")

#             if len(embeddings) != len(text_chunks):
#                 print(f"Mismatch: {len(embeddings)} embeddings vs {len(text_chunks)} text chunks")
#                 return False

#             collection = client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={"description": "Legal document embeddings"}
#             )

#             total_batches = (len(embeddings) + batch_size - 1) // batch_size
#             print(f"Total embeddings: {len(embeddings)} | Batch size: {batch_size}")

#             for batch_idx in range(total_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(embeddings))

#                 batch_embeddings = embeddings[start_idx:end_idx]
#                 batch_texts = text_chunks[start_idx:end_idx]
#                 batch_ids = [f"doc_{start_idx + i}" for i in range(len(batch_embeddings))]
#                 collection.add(
#                     embeddings=batch_embeddings.tolist(),
#                     documents=batch_texts,
#                     ids=batch_ids
#                 )
#                 print(f"Stored batch {batch_idx + 1}/{total_batches}")

#             print("All embeddings stored successfully.")
#             return True

#         except Exception as e:
#             print(f"Error in ChromaDB storage: {e}")
#             traceback.print_exc()
#             return False


# # === Example Usage ===
# if __name__ == "__main__":
#     print("=== Loading and Splitting Text ===")
#     split_docs = load_and_split_text("sample_text.txt", chunk_size=1000, chunk_overlap=200)

#     if not split_docs:
#         print("WARNING: load_and_split_text returned empty list. Using fallback text.")
#         sample_texts = [
#             "This is the first legal document chunk about contract law.",
#             "The second chunk discusses tort liability and damages.",
#             "This third chunk covers constitutional law principles."
#         ]
#         text_chunks = sample_texts
#     else:
#         text_chunks = [doc.page_content for doc in split_docs if doc and doc.page_content.strip()]

#     if not text_chunks:
#         raise ValueError("No valid text chunks to embed.")

#     embedder = LegalEmbedder()
#     embeddings = embedder.embed_texts(text_chunks)
#     output_file = embedder.save_embeddings(embeddings, "sample_embeddings.npy")

#     success = embedder.store_embeddings_in_chroma(
#         npy_file_path=output_file,
#         text_chunks=text_chunks,
#         collection_name="sample_documents",
#         batch_size=1000
#     )

#     if success:
#         print("=== ChromaDB Embedding Preview ===")
#         try:
#             client = chromadb.PersistentClient(path="./chroma_db")
#             collection = client.get_collection(name="sample_documents")

#             results = collection.get(include=["embeddings", "documents"])

#             if results \
#             and results.get("documents") and len(results.get("documents")) > 0 \
#             and results.get("embeddings") is not None and len(results.get("embeddings")) > 0:
    
#                 total = len(results['documents'])   
#                 print(f"Total documents stored in ChromaDB: {total}")
#                 print(f"Embedding dimension: {len(results['embeddings'][0])}")

#                 for i in range(min(5, total)):
#                     print(f"\nDoc {i+1} ID: {results['ids'][i]}")
#                     print(f"Text: {results['documents'][i][:300]}...")
#                     print(f"Embedding Preview: {results['embeddings'][i][:10]}...")
#             else:
#                 print("No documents or embeddings found in ChromaDB collection.")

#         except Exception as e:
#             print(f"Failed to preview ChromaDB contents: {e}")
#             traceback.print_exc()

#     else:
#         print("Failed to store embeddings in ChromaDB.")

#     try:
#         loaded_embeddings = np.load(output_file)
#         print(f"Verification: Loaded embeddings shape: {loaded_embeddings.shape}")
#     except Exception as e:
#         print(f"Error loading saved embeddings: {e}")
#         traceback.print_exc()















# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from typing import List
# from Loader import load_and_split_text
# import torch
# import chromadb
# import traceback

# class LegalEmbedder:
#     def __init__(self, model_name: str = "law-ai/InLegalBERT"):
#         print(f"Loading model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         print("Model loaded successfully!")

#     def embed_texts(self, texts: List[str]) -> np.ndarray:
#         if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
#             raise ValueError("Input to embed_texts must be a non-empty list of non-empty strings.")
        
#         print(f"Generating embeddings for {len(texts)} text chunks...")
#         embeddings_list = []

#         for i, text in enumerate(texts):
#             try:
#                 encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#                 with torch.no_grad():
#                     output = self.model(**encoded_input)
#                     last_hidden_state = output.last_hidden_state
#                 embedding = torch.mean(last_hidden_state, dim=1)
#                 embeddings_list.append(embedding.squeeze().numpy())
#             except Exception as e:
#                 print(f"Error processing text chunk {i}: {e}")
#                 traceback.print_exc()
#                 continue

#         if not embeddings_list:
#             raise RuntimeError("No valid embeddings were generated.")

#         embeddings_array = np.array(embeddings_list)
#         print(f"Generated embeddings shape: {embeddings_array.shape}")
#         return embeddings_array

#     def save_embeddings(self, embeddings: np.ndarray, output_path: str):
#         if not output_path.endswith('.npy'):
#             output_path += '.npy'
#         np.save(output_path, embeddings)
#         print(f"Embeddings saved to: {output_path}")
#         return output_path

#     def store_embeddings_in_chroma(self, npy_file_path: str, text_chunks: List[str], 
#                                    collection_name: str = "legal_documents", 
#                                    batch_size: int = 5000, 
#                                    persist_directory: str = "./chroma_db") -> bool:
#         try:
#             client = chromadb.PersistentClient(path=persist_directory)
#             print(f"ChromaDB client initialized at: {persist_directory}")

#             embeddings = np.load(npy_file_path)
#             print(f"Loaded embeddings from: {npy_file_path}, shape: {embeddings.shape}")

#             if len(embeddings) != len(text_chunks):
#                 print(f"Mismatch: {len(embeddings)} embeddings vs {len(text_chunks)} text chunks")
#                 return False

#             collection = client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={"description": "Legal document embeddings"}
#             )

#             total_batches = (len(embeddings) + batch_size - 1) // batch_size
#             print(f"Total embeddings: {len(embeddings)} | Batch size: {batch_size}")

#             for batch_idx in range(total_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(embeddings))

#                 batch_embeddings = embeddings[start_idx:end_idx]
#                 batch_texts = text_chunks[start_idx:end_idx]
#                 batch_ids = [f"doc_{start_idx + i}" for i in range(len(batch_embeddings))]
#                 collection.add(
#                     embeddings=batch_embeddings.tolist(),
#                     documents=batch_texts,
#                     ids=batch_ids
#                 )
#                 print(f"Stored batch {batch_idx + 1}/{total_batches}")

#             print("All embeddings stored successfully.")
#             return True

#         except Exception as e:
#             print(f"Error in ChromaDB storage: {e}")
#             traceback.print_exc()
#             return False


# # === Example Usage ===
# if __name__ == "__main__":
#     print("=== Loading and Splitting Text ===")
#     split_docs = load_and_split_text("sample_text.txt", chunk_size=1000, chunk_overlap=200)

#     if not split_docs:
#         print("WARNING: load_and_split_text returned empty list. Using fallback text.")
#         sample_texts = [
#             "This is the first legal document chunk about contract law.",
#             "The second chunk discusses tort liability and damages.",
#             "This third chunk covers constitutional law principles."
#         ]
#         text_chunks = sample_texts
#     else:
#         text_chunks = [doc.page_content for doc in split_docs if doc and doc.page_content.strip()]

#     if not text_chunks:
#         raise ValueError("No valid text chunks to embed.")

#     embedder = LegalEmbedder()
#     embeddings = embedder.embed_texts(text_chunks)
#     output_file = embedder.save_embeddings(embeddings, "sample_embeddings.npy")

#     success = embedder.store_embeddings_in_chroma(
#         npy_file_path=output_file,
#         text_chunks=text_chunks,
#         collection_name="sample_documents",
#         batch_size=1000
#     )

#     if success:
#         print("=== ChromaDB Embedding Preview ===")
#         try:
#             client = chromadb.PersistentClient(path="./chroma_db")
#             collection = client.get_collection(name="sample_documents")

#             # 🧠 FIXED: Include embeddings explicitly

#             results = collection.get(include=["embeddings", "documents"])

#             if results \
#             and results.get("documents") and len(results.get("documents")) > 0 \
#             and results.get("embeddings") is not None and len(results.get("embeddings")) > 0:
    
#                 total = len(results['documents'])   
#                 print(f"Total documents stored in ChromaDB: {total}")
#                 print(f"Embedding dimension: {len(results['embeddings'][0])}")

#                 for i in range(min(5, total)):
#                     print(f"\nDoc {i+1} ID: {results['ids'][i]}")
#                     print(f"Text: {results['documents'][i][:300]}...")
#                     print(f"Embedding Preview: {results['embeddings'][i][:10]}...")
#             else:
#                 print("No documents or embeddings found in ChromaDB collection.")


#             results = collection.get(include=["embeddings", "documents"])

#             if results and results.get("documents") and results.get("embeddings"):
#                 total = len(results['documents'])
#                 print(f"Total documents stored in ChromaDB: {total}")
#                 print(f"Embedding dimension: {len(results['embeddings'][0]) if results['embeddings'] else 'N/A'}")

#                 for i in range(min(5, total)):
#                     print(f"\nDoc {i+1} ID: {results['ids'][i]}")
#                     print(f"Text: {results['documents'][i][:300]}...")
#                     print(f"Embedding Preview: {results['embeddings'][i][:10]}...")
#             else:
#                 print("No documents or embeddings found in ChromaDB collection.")

#         except Exception as e:
#             print(f"Failed to preview ChromaDB contents: {e}")
#             traceback.print_exc()

#     else:
#         print("Failed to store embeddings in ChromaDB.")

#     try:
#         loaded_embeddings = np.load(output_file)
#         print(f"Verification: Loaded embeddings shape: {loaded_embeddings.shape}")
#     except Exception as e:
#         print(f"Error loading saved embeddings: {e}")
#         traceback.print_exc()















# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from typing import List
# from Loader import load_and_split_text
# import torch
# import chromadb

# class LegalEmbedder:
#     """
#     A simple embedder for legal text using law-ai/InLegalBERT model
#     """
    
#     def __init__(self, model_name: str = "law-ai/InLegalBERT"):
#         """
#         Initialize the embedder with the specified model
        
#         Args:
#             model_name: HuggingFace model name (default: law-ai/InLegalBERT)
#         """
#         print(f"Loading model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         print("Model loaded successfully!")
    
#     def embed_texts(self, texts: List[str]) -> np.ndarray:
#         """
#         Generate embeddings for a list of text chunks
        
#         Args:
#             texts: List of text strings to embed
            
#         Returns:
#             NumPy array of embeddings with shape (num_texts, embedding_dim)
#         """
#         print(f"Generating embeddings for {len(texts)} text chunks...")
        
#         embeddings_list = []
        
#         for text in texts:
#             # Tokenize the text
#             encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
#             # Get model output
#             with torch.no_grad():
#                 output = self.model(**encoded_input)
#                 last_hidden_state = output.last_hidden_state
            
#             # Use mean pooling to get sentence embedding
#             # Average across the sequence length dimension (dim=1)
#             embedding = torch.mean(last_hidden_state, dim=1)
            
#             # Convert to numpy and add to list
#             embeddings_list.append(embedding.squeeze().numpy())
        
#         # Convert list to numpy array
#         embeddings_array = np.array(embeddings_list)
#         print(f"Generated embeddings shape: {embeddings_array.shape}")
        
#         return embeddings_array
    
#     def save_embeddings(self, embeddings: np.ndarray, output_path: str):
#         """
#         Save embeddings to a .npy file
        
#         Args:
#             embeddings: NumPy array of embeddings
#             output_path: Path to save the .npy file
#         """
#         # Ensure output path has .npy extension
#         if not output_path.endswith('.npy'):
#             output_path += '.npy'
            
#         np.save(output_path, embeddings)
#         print(f"Embeddings saved to: {output_path}")
#         return output_path
    
#     def store_embeddings_in_chroma(self, npy_file_path: str, text_chunks: List[str], 
#                                   collection_name: str = "legal_documents", 
#                                   batch_size: int = 5000, 
#                                   persist_directory: str = "./chroma_db") -> bool:
#         """
#         Load embeddings from .npy file and store them in ChromaDB with batch processing
        
#         Args:
#             npy_file_path: Path to the .npy file (output from save_embeddings)
#             text_chunks: List of corresponding text chunks
#             collection_name: Name of the ChromaDB collection
#             batch_size: Number of embeddings to process in each batch
#             persist_directory: Directory to persist ChromaDB data
            
#         Returns:
#             True if successful, False otherwise
#         """
#         try:
#             # Initialize ChromaDB client
#             client = chromadb.PersistentClient(path=persist_directory)
#             print(f"ChromaDB client initialized with persist directory: {persist_directory}")
            
#             # Load embeddings from .npy file
#             embeddings = np.load(npy_file_path)
#             print(f"Loaded embeddings from {npy_file_path}")
#             print(f"Embeddings shape: {embeddings.shape}")
            
#             # Verify that embeddings and texts have the same length
#             if len(embeddings) != len(text_chunks):
#                 print(f"Error: Number of embeddings ({len(embeddings)}) doesn't match number of texts ({len(text_chunks)})")
#                 return False
            
#             # Get or create collection
#             collection = client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={"description": "Legal document embeddings"}
#             )
            
#             print(f"Using collection: {collection_name}")
#             print(f"Total embeddings to store: {len(embeddings)}")
#             print(f"Batch size: {batch_size}")
            
#             # Process embeddings in batches
#             total_batches = (len(embeddings) + batch_size - 1) // batch_size
            
#             for batch_idx in range(total_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(embeddings))
                
#                 batch_embeddings = embeddings[start_idx:end_idx]
#                 batch_texts = text_chunks[start_idx:end_idx]
                
#                 # Create IDs for this batch
#                 batch_ids = [f"doc_{start_idx + i}" for i in range(len(batch_embeddings))]
                
#                 # Convert embeddings to list format for ChromaDB
#                 batch_embeddings_list = batch_embeddings.tolist()
                
#                 # Add documents to collection
#                 collection.add(
#                     embeddings=batch_embeddings_list,
#                     documents=batch_texts,
#                     ids=batch_ids
#                 )
                
#                 print(f"Stored batch {batch_idx + 1}/{total_batches} ({len(batch_embeddings)} embeddings)")
            
#             print(f"Successfully stored all {len(embeddings)} embeddings in ChromaDB")
#             return True
            
#         except Exception as e:
#             print(f"Error storing embeddings in ChromaDB: {e}")
#             return False

# # Example usage
# if __name__ == "__main__":
#     # Load and split the sample text file using the existing function from Loader.py
#     print("Attempting to load and split sample_text.txt...")
#     split_docs = load_and_split_text("sample_text.txt", chunk_size=1000, chunk_overlap=200)
    
#     if not split_docs:
#         print("WARNING: load_and_split_text returned None. This could be due to:")
#         print("1. Missing sample_text.txt file")
#         print("2. LangChain dependencies not installed")
#         print("3. File encoding issues")
#         print("4. Other file loading errors")
#         print("\nUsing fallback sample texts...")
#         sample_texts = [
#             "This is the first legal document chunk about contract law.",
#             "The second chunk discusses tort liability and damages.",
#             "This third chunk covers constitutional law principles."
#         ]
#         text_chunks = sample_texts
#     else:
#         print(f"Successfully loaded {len(split_docs)} document chunks")
#         # Extract the page_content from the Document objects
#         text_chunks = [doc.page_content for doc in split_docs]
    
#     # Initialize embedder
#     embedder = LegalEmbedder()
    
#     # Generate embeddings
#     embeddings = embedder.embed_texts(text_chunks)
    
#     # Save to file
#     output_file = embedder.save_embeddings(embeddings, "sample_embeddings.npy")
    
#     # Store embeddings in ChromaDB
#     success = embedder.store_embeddings_in_chroma(
#         npy_file_path=output_file,
#         text_chunks=text_chunks,
#         collection_name="sample_documents",
#         batch_size=1000
#     )
    
#     if success:
#         print("Successfully stored embeddings in ChromaDB!")
        
#         # Display first 5 embeddings stored in ChromaDB along with their original text
#         print("\n" + "="*80)
#         print("FIRST 5 EMBEDDINGS STORED IN CHROMADB")
#         print("="*80)
        
#         try:
#             # Initialize ChromaDB client to query the stored data
#             client = chromadb.PersistentClient(path="./chroma_db")
#             collection = client.get_collection(name="sample_documents")
            
#             # Get all documents (we'll limit to first 5 in display)
#             results = collection.get()
            
#             if results and 'documents' in results and len(results['documents']) > 0:
#                 print(f"Total documents stored in ChromaDB: {len(results['documents'])}")
#                 print(f"Total embeddings stored: {len(results['embeddings'])}")
#                 print(f"Embedding dimension: {len(results['embeddings'][0])}")
#                 print()
                
#                 # Display first 5 embeddings and their text
#                 for i in range(min(5, len(results['documents']))):
#                     print(f"Document {i+1} (ID: {results['ids'][i]}):")
#                     print(f"Text: {results['documents'][i][:300]}...")
#                     print(f"Embedding (first 10 values): {results['embeddings'][i][:10]}...")
#                     print(f"Embedding shape: {len(results['embeddings'][i])} dimensions")
#                     print("-" * 80)
#             else:
#                 print("No documents found in ChromaDB collection")
                
#         except Exception as e:
#             print(f"Error retrieving data from ChromaDB: {e}")
#     else:
#         print("Failed to store embeddings in ChromaDB")
    
#     # Verify the saved file
#     loaded_embeddings = np.load(output_file)
#     print(f"Verification: Loaded embeddings shape: {loaded_embeddings.shape}")