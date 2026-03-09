import os
from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PINECONE_ENV
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define index name
INDEX_NAME = "media-talent"

class VectorDBClient:
    def __init__(self):
        """Initialize Pinecone connection."""
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing from .env")
            
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index_exists()
        self.index = self.pc.Index(INDEX_NAME)

        # For text embeddings
        self.embeddings = HuggingFaceEmbeddings(
            # Using a fast, free, local sentence transformer
            model_name="sentence-transformers/all-MiniLM-L6-v2" 
        )

    def _ensure_index_exists(self):
        """Creates the index if it doesn't already exist or if dimensions mismatch."""
        import time
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        recreate = False
        if INDEX_NAME in existing_indexes:
            # Check dimension of existing index
            desc = self.pc.describe_index(INDEX_NAME)
            if desc.dimension != 384:
                print(f"Dimension mismatch for '{INDEX_NAME}' (Found {desc.dimension}, Need 384). Recreating...")
                self.pc.delete_index(INDEX_NAME)
                time.sleep(5) # Small buffer for deletion to propagate
                recreate = True
            else:
                print(f"Index '{INDEX_NAME}' already exists with correct dimensions.")
        else:
            recreate = True

        if recreate:
            print(f"Creating new Pinecone index: '{INDEX_NAME}'...")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=384, # all-MiniLM-L6-v2 outputs 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for readiness
            print("Waiting for index to be ready...")
            while not self.pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(2)
            print("Index created and ready.")

    def get_index(self):
        return self.index

if __name__ == "__main__":
    # Test connection
    client = VectorDBClient()
    print("VectorDB initialized.", client.get_index().describe_index_stats())
