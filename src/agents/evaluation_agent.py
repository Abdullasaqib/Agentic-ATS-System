from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class EvaluationAgent:
    def __init__(self):
        # Initialize the LLM for reasoning
        # Using LLaMA 3 8b via Groq for speed
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
        
        # Prompt for evaluation
        self.eval_prompt = PromptTemplate(
            input_variables=["role", "vibe", "candidate_text"],
            template="""
            You are a Senior Executive Recruiter at H&H International Strategy (H&HIS), specializing in the Media and Entertainment industry.
            Your job is to provide a "Red Carpet" evaluation of a candidate based on their profile.
            
            Job Role: {role}
            Desired Production Style / Cultural Vibe: {vibe}
            
            Candidate Data:
            {candidate_text}
            
            Provide a sophisticated, punchy 3-4 sentence professional evaluation. 
            Focus specifically on:
            1. Production Style Alignment: Does their aesthetic/technical background match the requested vibe?
            2. Cultural Fit: Do they 'speak the language' of this specific production role?
            3. Final Recommendation: A clear "High Potential", "Possible Fit", or "Low Alignment" summary.
            
            Professional Evaluation:
            """
        )

    def evaluate_candidate(self, role: str, vibe: str, candidate_text: str) -> str:
        """
        Takes a sourced candidate's text and reasons about their fit.
        """
        chain = self.eval_prompt | self.llm
        
        response = chain.invoke({
            "role": role,
            "vibe": vibe,
            "candidate_text": candidate_text
        })
        
        return response.content

if __name__ == "__main__":
    agent = EvaluationAgent()
    sample_text = "Name: Alex River. Role: Documentary Producer. Experience: 10 years. Known for gritty, character-driven storytelling."
    reasoning = agent.evaluate_candidate("Documentary Producer", "Gritty, character-driven, narrative focused.", sample_text)
    print("Evaluation:", reasoning)
