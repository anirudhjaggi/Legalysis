"""
Main Orchestrator for RAG Legal Chatbot

This script runs all 4 components in sequence:
1. Loader.py - Load and split text documents
2. Embedder.py - Generate embeddings and store in ChromaDB
3. Retriever.py - Create retriever for document search
4. LLM.py - Create chatbot with Google Gemini integration

Usage:
    python main.py
"""

import os
import sys
from typing import Dict, Any, Optional
import traceback

# Import our modules
from Utils.Loader import load_and_split_text
from Utils.Embedder import LegalEmbedder
from Utils.Retriever import Retriever
from Utils.LLM import RAGChatbot


class LegalChatbotOrchestrator:
    """
    Main orchestrator class that manages the entire RAG pipeline
    """
    
    def __init__(self):
        self.config = self.get_default_config()
        self.documents = None
        self.embedder = None
        self.retriever = None
        self.chatbot = None
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with all customizable parameters"""
        return {
            # File paths
            "input_file": "sample_text.txt",
            "output_embeddings_file": "sample_embeddings.npy",
            "chroma_db_directory": "./Sample_chroma_db",
            
            # Text processing
            "chunk_size": 1000,
            "chunk_overlap": 200,
            
            # Embedding model
            "embedding_model": "law-ai/InLegalBERT",
            
            # ChromaDB settings
            "collection_name": "sample_documents",
            "batch_size": 1000,
            
            # LLM settings
            "llm_model": "gemini-1.5-flash",
            "temperature": 0.5,
            "max_tokens": 10000,
            
            # Retrieval settings
            "retrieval_k": 4,
            
            # Processing flags
            "skip_embedding_generation": False,
            "skip_chroma_storage": False,
            "start_chat_immediately": True
        }
    
    def customize_config(self) -> Dict[str, Any]:
        """Allow user to customize configuration"""
        print("\n" + "="*60)
        print("🤖 LEGAL CHATBOT CONFIGURATION")
        print("="*60)
        
        config = self.config.copy()
        
        # File paths
        print("\n📁 FILE PATHS:")
        config["input_file"] = input(f"Input text file (default: {config['input_file']}): ").strip() or config["input_file"]
        config["output_embeddings_file"] = input(f"Output embeddings file (default: {config['output_embeddings_file']}): ").strip() or config["output_embeddings_file"]
        config["chroma_db_directory"] = input(f"ChromaDB directory (default: {config['chroma_db_directory']}): ").strip() or config["chroma_db_directory"]
        
        # Text processing
        print("\n📝 TEXT PROCESSING:")
        try:
            chunk_size = input(f"Chunk size (default: {config['chunk_size']}): ").strip()
            if chunk_size:
                config["chunk_size"] = int(chunk_size)
        except ValueError:
            print("Invalid chunk size, using default")
            
        try:
            chunk_overlap = input(f"Chunk overlap (default: {config['chunk_overlap']}): ").strip()
            if chunk_overlap:
                config["chunk_overlap"] = int(chunk_overlap)
        except ValueError:
            print("Invalid chunk overlap, using default")
        
        # Model settings
        print("\n🧠 MODEL SETTINGS:")
        config["embedding_model"] = input(f"Embedding model (default: {config['embedding_model']}): ").strip() or config["embedding_model"]
        config["llm_model"] = input(f"LLM model (default: {config['llm_model']}): ").strip() or config["llm_model"]
        
        try:
            temperature = input(f"Temperature (default: {config['temperature']}): ").strip()
            if temperature:
                config["temperature"] = float(temperature)
        except ValueError:
            print("Invalid temperature, using default")
            
        try:
            max_tokens = input(f"Max tokens (default: {config['max_tokens']}): ").strip()
            if max_tokens:
                config["max_tokens"] = int(max_tokens)
        except ValueError:
            print("Invalid max tokens, using default")
        
        # ChromaDB settings
        print("\n🗄️ CHROMADB SETTINGS:")
        config["collection_name"] = input(f"Collection name (default: {config['collection_name']}): ").strip() or config["collection_name"]
        
        try:
            batch_size = input(f"Batch size (default: {config['batch_size']}): ").strip()
            if batch_size:
                config["batch_size"] = int(batch_size)
        except ValueError:
            print("Invalid batch size, using default")
        
        # Processing options
        print("\n⚙️ PROCESSING OPTIONS:")
        skip_embedding = input("Skip embedding generation? (y/N): ").strip().lower() == 'y'
        config["skip_embedding_generation"] = skip_embedding
        
        skip_chroma = input("Skip ChromaDB storage? (y/N): ").strip().lower() == 'y'
        config["skip_chroma_storage"] = skip_chroma
        
        start_chat = input("Start chat immediately? (Y/n): ").strip().lower()
        config["start_chat_immediately"] = start_chat != 'n'
        
        return config
    
    def step_1_load_documents(self) -> bool:
        """Step 1: Load and split documents"""
        print("\n" + "="*50)
        print("📚 STEP 1: LOADING DOCUMENTS")
        print("="*50)
        
        try:
            if not os.path.exists(self.config["input_file"]):
                print(f"❌ Error: Input file '{self.config['input_file']}' not found!")
                return False
            
            print(f"📖 Loading file: {self.config['input_file']}")
            print(f"🔧 Chunk size: {self.config['chunk_size']}, Overlap: {self.config['chunk_overlap']}")
            
            self.documents = load_and_split_text(
                self.config["input_file"],
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            
            if not self.documents:
                print("❌ Error: No documents loaded!")
                return False
            
            print(f"✅ Successfully loaded {len(self.documents)} document chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error in document loading: {e}")
            traceback.print_exc()
            return False
    
    def step_2_generate_embeddings(self) -> bool:
        """Step 2: Generate embeddings"""
        print("\n" + "="*50)
        print("🧠 STEP 2: GENERATING EMBEDDINGS")
        print("="*50)
        
        if self.config["skip_embedding_generation"]:
            print("⏭️ Skipping embedding generation as requested")
            return True
        
        try:
            if not self.documents:
                print("❌ Error: No documents available for embedding!")
                return False
            
            text_chunks = [doc.page_content for doc in self.documents if doc and doc.page_content.strip()]
            
            if not text_chunks:
                print("❌ Error: No valid text chunks to embed!")
                return False
            
            print(f"🔧 Using model: {self.config['embedding_model']}")
            print(f"📝 Processing {len(text_chunks)} text chunks...")
            
            self.embedder = LegalEmbedder(model_name=self.config["embedding_model"])
            embeddings = self.embedder.embed_texts(text_chunks)
            
            # Save embeddings
            output_file = self.embedder.save_embeddings(embeddings, self.config["output_embeddings_file"])
            self.config["output_embeddings_file"] = output_file
            
            print(f"✅ Embeddings generated and saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error in embedding generation: {e}")
            traceback.print_exc()
            return False
    
    def step_3_store_in_chromadb(self) -> bool:
        """Step 3: Store embeddings in ChromaDB"""
        print("\n" + "="*50)
        print("🗄️ STEP 3: STORING IN CHROMADB")
        print("="*50)
        
        if self.config["skip_chroma_storage"]:
            print("⏭️ Skipping ChromaDB storage as requested")
            return True
        
        try:
            if not self.embedder or not self.documents:
                print("❌ Error: Embedder or documents not available!")
                return False
            
            text_chunks = [doc.page_content for doc in self.documents if doc and doc.page_content.strip()]
            
            print(f"📦 Storing in ChromaDB directory: {self.config['chroma_db_directory']}")
            print(f"📚 Collection name: {self.config['collection_name']}")
            print(f"🔧 Batch size: {self.config['batch_size']}")
            
            success = self.embedder.store_embeddings_in_chroma(
                npy_file_path=self.config["output_embeddings_file"],
                text_chunks=text_chunks,
                collection_name=self.config["collection_name"],
                batch_size=self.config["batch_size"],
                persist_directory=self.config["chroma_db_directory"]
            )
            
            if success:
                print("✅ Successfully stored embeddings in ChromaDB")
                return True
            else:
                print("❌ Failed to store embeddings in ChromaDB")
                return False
                
        except Exception as e:
            print(f"❌ Error in ChromaDB storage: {e}")
            traceback.print_exc()
            return False
    
    def step_4_create_retriever(self) -> bool:
        """Step 4: Create retriever"""
        print("\n" + "="*50)
        print("🔍 STEP 4: CREATING RETRIEVER")
        print("="*50)
        
        try:
            if not os.path.exists(self.config["chroma_db_directory"]):
                print(f"❌ Error: ChromaDB directory '{self.config['chroma_db_directory']}' not found!")
                return False
            
            print(f"🔧 Using model: {self.config['embedding_model']}")
            print(f"📚 Collection: {self.config['collection_name']}")
            print(f"📁 Directory: {self.config['chroma_db_directory']}")
            
            self.retriever = Retriever(
                persist_directory=self.config["chroma_db_directory"],
                collection_name=self.config["collection_name"],
                model_name=self.config["embedding_model"]
            )
            
            print("✅ Retriever created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating retriever: {e}")
            traceback.print_exc()
            return False
    
    def step_5_create_chatbot(self) -> bool:
        """Step 5: Create chatbot"""
        print("\n" + "="*50)
        print("🤖 STEP 5: CREATING CHATBOT")
        print("="*50)
        
        try:
            if not self.retriever:
                print("❌ Error: Retriever not available!")
                return False
            
            print(f"🔧 LLM model: {self.config['llm_model']}")
            print(f"🌡️ Temperature: {self.config['temperature']}")
            print(f"📏 Max tokens: {self.config['max_tokens']}")
            
            self.chatbot = RAGChatbot(
                retriever=self.retriever,
                model_name=self.config["llm_model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            
            print("✅ Chatbot created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating chatbot: {e}")
            traceback.print_exc()
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        print("\n🚀 STARTING LEGAL CHATBOT PIPELINE")
        print("="*60)
        
        # Customize configuration
        self.config = self.customize_config()
        
        # Step 1: Load documents
        if not self.step_1_load_documents():
            return False
        
        # Step 2: Generate embeddings
        if not self.step_2_generate_embeddings():
            return False
        
        # Step 3: Store in ChromaDB
        if not self.step_3_store_in_chromadb():
            return False
        
        # Step 4: Create retriever
        if not self.step_4_create_retriever():
            return False
        
        # Step 5: Create chatbot
        if not self.step_5_create_chatbot():
            return False
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Start chat if requested
        if self.config["start_chat_immediately"]:
            print("\n💬 Starting interactive chat...")
            self.chatbot.chat()
        
        return True
    
    def test_retrieval(self, query: str = "What is contract law?", k: int = 3):
        """Test the retriever with a sample query"""
        print(f"\n🧪 TESTING RETRIEVAL")
        print(f"Query: {query}")
        print(f"Retrieving top {k} results...")
        
        try:
            results = self.retriever.retrieve(query, k=k)
            print(f"\n📋 Found {len(results)} results:")
            
            for i, doc in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"ID: {doc['id']}")
                print(f"Score: {doc['score']:.4f}")
                print(f"Content: {doc['document'][:200]}...")
                
        except Exception as e:
            print(f"❌ Error testing retrieval: {e}")
            traceback.print_exc()
    
    def test_chatbot(self, question: str = "What is contract law?"):
        """Test the chatbot with a sample question"""
        print(f"\n🧪 TESTING CHATBOT")
        print(f"Question: {question}")
        
        try:
            response = self.chatbot.query(question, k=self.config["retrieval_k"])
            print(f"\n🤖 Answer: {response['answer']}")
            print(f"📚 Sources used: {response['num_sources']}")
            
            if response['num_sources'] > 0:
                print("\n📄 Source previews:")
                for i, src in enumerate(response['sources'][:2], 1):
                    print(f"Source {i}: {src['content'][:150]}...")
                    
        except Exception as e:
            print(f"❌ Error testing chatbot: {e}")
            traceback.print_exc()


def main():
    """Main function to run the complete pipeline"""
    print("🤖 LEGAL CHATBOT - COMPLETE PIPELINE")
    print("="*60)
    print("This script will run all 4 components:")
    print("1. 📚 Loader - Load and split text documents")
    print("2. 🧠 Embedder - Generate embeddings")
    print("3. 🔍 Retriever - Create document retriever")
    print("4. 🤖 LLM - Create chatbot with Google Gemini")
    print("="*60)
    
    try:
        orchestrator = LegalChatbotOrchestrator()
        
        # Run the complete pipeline
        success = orchestrator.run_pipeline()
        
        if success:
            print("\n🎯 PIPELINE COMPLETED!")
            
            # Ask if user wants to test
            test_choice = input("\n🧪 Would you like to test the system? (y/N): ").strip().lower()
            
            if test_choice == 'y':
                print("\n" + "="*50)
                print("🧪 TESTING MODE")
                print("="*50)
                
                # Test retrieval
                test_query = input("Enter test query (default: 'What is contract law?'): ").strip()
                if not test_query:
                    test_query = "What is contract law?"
                
                orchestrator.test_retrieval(test_query)
                
                # Test chatbot
                test_question = input("\nEnter test question (default: 'What is contract law?'): ").strip()
                if not test_question:
                    test_question = "What is contract law?"
                
                orchestrator.test_chatbot(test_question)
                
                # Ask if user wants to start chat
                chat_choice = input("\n💬 Start interactive chat? (Y/n): ").strip().lower()
                if chat_choice != 'n':
                    print("\n💬 Starting interactive chat...")
                    orchestrator.chatbot.chat()
        
        else:
            print("\n❌ PIPELINE FAILED!")
            print("Please check the error messages above and try again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 