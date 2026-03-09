import streamlit as st
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.sourcing_agent import SourcingAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.interview_agent import InterviewAgent
from src.ingestion import BaseIngestor

from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

def get_agents():
    """Returns the agents. Removed caching to ensure LLM model updates are applied immediately."""
    return SourcingAgent(), EvaluationAgent(), InterviewAgent()

@st.cache_resource
def get_ingestor():
    return BaseIngestor()

def main():
    st.set_page_config(page_title="H&HIS Media Talent AI", page_icon="🎬", layout="wide")
    
    st.title("🎬 Multi-Agent RAG: H&HIS Media Talent Command Center")
    st.write("Welcome to the H&H International Strategy talent orchestrator. Upload resumes and find the perfect creative fit.")

    try:
        sourcing_agent, eval_agent, interview_agent = get_agents()
        ingestor = get_ingestor()
    except Exception as e:
        st.error(f"Error initializing system (Check API keys): {e}")
        return

    # Sidebar for controls
    with st.sidebar:
        st.header("1. Upload New Talent")
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        
        if uploaded_file is not None:
            if st.button("Ingest Resume"):
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    file_content = uploaded_file.read()
                    file_type = uploaded_file.name.split(".")[-1]
                    cand_id = ingestor.process_and_upsert_resume(file_content, uploaded_file.name, file_type)
                    st.success(f"Successfully ingested {uploaded_file.name} as {cand_id}!")
        
        st.divider()
        st.header("2. Search & Match")
        job_role = st.text_input("Job Role", "Documentary Producer")
        required_vibe = st.text_area("Required Vibe / Style", "Gritty, character-driven storytelling.")
        
        start_run = st.button("Start AI Matching Run", type="primary")

        st.divider()
        if st.button("Clear System Cache (Fixes LLM Errors)"):
            st.cache_resource.clear()
            st.success("Cache cleared! Re-initializing agents...")
            st.rerun()

    # Layout
    col_candidates, col_logs = st.columns([1, 1])

    if start_run:
        with st.spinner("Agents are working..."):
            # 1. Sourcing Agent
            with col_logs:
                st.subheader("Agent Reasoning Log")
                log_box = st.empty()
                log_text = f"**[Sourcing Agent]** Searching Vector DB for '{job_role}' matching vibe '{required_vibe}'...\n"
                log_box.markdown(log_text)

            candidates = sourcing_agent.query_candidates(job_role, required_vibe, top_k=3)
            
            if not candidates:
                log_text += "\n**[Sourcing Agent]** No matching candidates found."
                log_box.markdown(log_text)
                with col_candidates:
                    st.warning("No candidates found in the database. Please upload a resume first.")
                return

            log_text += f"\n**[Sourcing Agent]** Found {len(candidates)} candidates. Filtering top matches...\n"
            log_box.markdown(log_text)

            # Display Candidates and run Evaluation/Interview pipeline
            with col_candidates:
                st.subheader("Candidate Shortlist & AI Workflow")
                
                for idx, candidate in enumerate(candidates):
                    with st.expander(f"Match #{idx+1}: {candidate['id']} (Score: {candidate['score']:.2f})", expanded=True):
                        st.markdown(f"**Chunk Content:**\n{candidate['text'][:500]}...")
                        
                        # 2. Evaluation Agent
                        log_text += f"\n**[Evaluation Agent]** Reviewing {candidate['id']}..."
                        log_box.markdown(log_text)
                        
                        reasoning = eval_agent.evaluate_candidate(
                            role=job_role, 
                            vibe=required_vibe, 
                            candidate_text=candidate['text']
                        )
                        st.info(f"**Headhunter AI Evaluation:**\n{reasoning}")
                        
                        # 3. Interview Agent
                        log_text += f"\n**[Interview Agent]** Generating questions for {candidate['id']}..."
                        log_box.markdown(log_text)
                        
                        questions = interview_agent.generate_questions(
                            role=job_role,
                            vibe=required_vibe,
                            candidate_text=candidate['text'],
                            evaluation_reasoning=reasoning
                        )
                        st.success(f"**Suggested Interview Questions:**\n{questions}")
                        
            log_text += "\n**[Orchestrator]** Run complete."
            log_box.markdown(log_text)
            
    else:
        with col_candidates:
            st.subheader("Candidate Shortlist Workflow")
            st.info("No run active. Upload a resume or click 'Start AI Matching Run' in the sidebar.")
        with col_logs:
            st.subheader("Agent Reasoning Log")
            st.code("System is idle.")

if __name__ == "__main__":
    main()
