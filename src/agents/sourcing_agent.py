from src.vector_db import VectorDBClient

class SourcingAgent:
    def __init__(self):
        self.db_client = VectorDBClient()
        self.index = self.db_client.get_index()
        self.embeddings = self.db_client.embeddings

    def query_candidates(self, role: str, vibe: str, top_k: int = 5):
        """
        Takes the role and desired 'vibe' to run a semantic search
        over the RAG vector DB. Groups chunks by candidate ID.
        """
        print(f"Sourcing Agent searching for: {role} with vibe: {vibe}")
        query_text = f"Role: {role}. Requirements: {vibe}"
        
        # Generate embedding for the search query
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Search the Pinecone index (return more chunks to allow for grouping)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 3, 
            include_metadata=True
        )
        
        # Group by candidate ID
        grouped_candidates = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            # Use the 'id' from metadata if available (set during ingestion), 
            # otherwise fall back to the vector id
            cand_id = metadata.get("id", match["id"].split("_chunk_")[0])
            
            if cand_id not in grouped_candidates:
                grouped_candidates[cand_id] = {
                    "id": cand_id,
                    "score": match["score"],
                    "text": metadata.get("text", ""),
                    "chunks": [metadata.get("text", "")]
                }
            else:
                # Append text from other chunks of the same candidate
                if metadata.get("text", "") not in grouped_candidates[cand_id]["chunks"]:
                    grouped_candidates[cand_id]["chunks"].append(metadata.get("text", ""))
                    # Keep the highest match score
                    grouped_candidates[cand_id]["score"] = max(grouped_candidates[cand_id]["score"], match["score"])

        # Final list of unique candidates with consolidated text
        final_candidates = []
        for cand in list(grouped_candidates.values()):
            cand["text"] = "\n---\n".join(cand["chunks"])
            final_candidates.append(cand)

        # Sort by score and limit to top_k
        final_candidates = sorted(final_candidates, key=lambda x: x["score"], reverse=True)[:top_k]
            
        print(f"Sourcing Agent found {len(final_candidates)} unique candidates.")
        return final_candidates

if __name__ == "__main__":
    agent = SourcingAgent()
    results = agent.query_candidates("Producer", "Gritty and narrative driven", top_k=2)
    for r in results:
        print(f"Score: {r['score']:.2f} | ID: {r['id']}")
