from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class InterviewAgent:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)
        
        self.interview_prompt = PromptTemplate(
            input_variables=["role", "vibe", "candidate_text", "evaluation_reasoning"],
            template="""
            You are the Lead Interviewer for a media production company.
            You are hiring for: {role}
            Looking for this style: {vibe}
            
            Candidate Profile:
            {candidate_text}
            
            Headhunter's Evaluation:
            {evaluation_reasoning}
            
            Based on the profile and evaluation, generate exactly 3 highly specific 
            interview questions to ask this candidate. The questions should probe 
            into their specific style and confirm if they truly fit the "vibe".
            
            Questions:
            """
        )

    def generate_questions(self, role: str, vibe: str, candidate_text: str, evaluation_reasoning: str) -> str:
        chain = self.interview_prompt | self.llm
        
        response = chain.invoke({
            "role": role,
            "vibe": vibe,
            "candidate_text": candidate_text,
            "evaluation_reasoning": evaluation_reasoning
        })
        
        return response.content
