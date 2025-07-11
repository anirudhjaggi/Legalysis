import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from Utils.Loader import load_and_split_text
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
