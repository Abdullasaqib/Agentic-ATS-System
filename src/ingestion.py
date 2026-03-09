import io
import docx2txt
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from src.vector_db import VectorDBClient

class ResumeProcessor:
    @staticmethod
    def extract_text(file_content: bytes, file_type: str) -> str:
        """Extracts text from PDF or DOCX file content."""
        if file_type == "pdf":
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif file_type == "docx":
            return docx2txt.process(io.BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

class BaseIngestor:
    def __init__(self):
        self.db_client = VectorDBClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_and_split(self, file_path: str):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def process_and_upsert_resume(self, file_content: bytes, file_name: str, file_type: str):
        """Processes an uploaded resume and upserts it into Pinecone. Cleans up old versions."""
        text = ResumeProcessor.extract_text(file_content, file_type)
        candidate_id = f"user_upload_{file_name.replace(' ', '_')}"
        
        # Clean up old vectors for this candidate to prevent duplicates
        print(f"Cleaning up old vectors for {candidate_id}...")
        try:
            self.db_client.index.delete(filter={"id": {"$eq": candidate_id}})
        except Exception as e:
            print(f"No existing vectors found or delete failed: {e}")

        chunks = self.text_splitter.split_text(text)
        embeds = self.db_client.embeddings.embed_documents(chunks)
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vectors_to_upsert.append({
                "id": f"{candidate_id}_chunk_{i}",
                "values": embeds[i],
                "metadata": {
                    "text": chunk,
                    "id": candidate_id,
                    "source": file_name,
                    "role": "Uploaded Candidate"
                }
            })

        self.db_client.index.upsert(vectors=vectors_to_upsert)
        return candidate_id

    def upsert_mock_candidates(self):
        """
        Upserts some mock candidate data to the DB for the sourcing agent to find.
        """
        print("Upserting mock candidates...")
        mock_candidates = [
            {
                "id": "cand_001",
                "text": "Name: Alex River. Role: Documentary Producer. Experience: 10 years. Known for gritty, character-driven storytelling. Worked on 3 indie films that won awards at Sundance. Strong background in narrative structuring and leading small crew sizes in remote locations. Excellent at finding the truth behind the story.",
                "metadata": {"role": "Producer", "years_experience": 10, "style": "Gritty, Narrative, Indie"}
            },
            {
                "id": "cand_002",
                "text": "Name: Sam Chen. Role: Commercial Videographer. Experience: 5 years. Specializes in high-gloss, fast-paced product commercials. Expert in lighting and slow-motion capture. Very corporate and clean aesthetic. Clients include Nike and Apple. Not well suited for unstructured storytelling.",
                "metadata": {"role": "Videographer", "years_experience": 5, "style": "Corporate, Glossy, Fast-paced"}
            },
            {
                "id": "cand_003",
                "text": "Name: Jordan Hayes. Role: Senior Podcast Editor. Experience: 8 years. Master of true-crime and investigative audio storytelling. Builds incredible tension through sound design. Has worked closely with investigative journalists to turn raw tape into compelling narratives.",
                "metadata": {"role": "Audio Editor", "years_experience": 8, "style": "Investigative, Tension, True-crime"}
            }
        ]

        # Convert to vector format
        texts = [c["text"] for c in mock_candidates]
        embeds = self.db_client.embeddings.embed_documents(texts)
        
        vectors_to_upsert = []
        for i, candidate in enumerate(mock_candidates):
            vectors_to_upsert.append({
                "id": candidate["id"],
                "values": embeds[i],
                "metadata": {
                    "text": candidate["text"],
                    **candidate["metadata"]
                }
            })

        self.db_client.index.upsert(vectors=vectors_to_upsert)
        print("Mock candidates upserted successfully.")

if __name__ == "__main__":
    # Test ingestion
    ingestor = BaseIngestor()
    ingestor.upsert_mock_candidates()
