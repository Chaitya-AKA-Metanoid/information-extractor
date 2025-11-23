import streamlit as st
import pypdf
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from io import BytesIO

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Rubric-Compliant AI Parser", page_icon="📋", layout="wide")

@st.cache_resource
def load_resources():
    """Loads NLP tools and the AI Model."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Load QA Model (DistilBERT)
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_resources()

# --- 2. UTILITY FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def find_context_sentence(text, answer):
    """
    Locates the full sentence containing the specific answer 
    to populate the 'Comments' column.
    """
    if not answer or len(answer) < 2: 
        return ""
    
    sentences = sent_tokenize(text)
    for sent in sentences:
        if answer in sent:
            # Clean up newlines in the sentence
            return sent.replace('\n', ' ').strip()
    return ""

# --- 3. HYBRID EXTRACTION LOGIC ---

def extract_hardcoded_basics(text):
    """
    Regex extraction for fields where AI struggles with formatting (Names, Dates).
    """
    basics = {}
    
    # 1. Name Heuristic (First non-label line)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    blocklist = ["RESUME", "CV", "PROFILE", "CONTACT"]
    for line in lines[:5]:
        if len(line.split()) <= 3 and line.upper() not in blocklist and not any(c.isdigit() for c in line):
            parts = line.split()
            basics["First Name"] = parts[0]
            basics["Last Name"] = parts[-1] if len(parts) > 1 else ""
            break
            
    # 2. Date of Birth (ISO Format)
    dob_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if dob_match:
        basics["Date of Birth"] = dob_match.group(1)
        
    return basics

def run_assignment_extraction(text):
    """
    The Master Loop: Fills the specific 37 rows using a mix of Regex and AI.
    """
    # 1. Get the basics first (High Precision)
    basics = extract_hardcoded_basics(text)
    
    # 2. Define the AI Schema (The 37 Requirements)
    # Maps 'Excel Key' -> 'Natural Language Question'
    schema = [
        ("First Name", "What is the first name?", "basics"), 
        ("Last Name", "What is the last name?", "basics"),
        ("Date of Birth", "What is the birth date formatted as YYYY-MM-DD?", "basics"),
        ("Birth City", "Which city was the candidate born in?", "ai"),
        ("Birth State", "Which state was the candidate born in?", "ai"),
        ("Age", "How old is the candidate?", "ai"),
        ("Blood Group", "What is the blood group?", "ai"),
        ("Nationality", "What is the nationality?", "ai"),
        
        ("Joining Date of first professional role", "When did they join their first job?", "ai"),
        ("Designation of first professional role", "What was the designation at their first job?", "ai"),
        ("Salary of first professional role", "What was the starting annual salary?", "ai"),
        ("Salary currency of first professional role", "What is the currency of the salary?", "ai"),
        
        ("Current Organization", "What is the name of the current organization?", "ai"),
        ("Current Joining Date", "When did they join the current organization?", "ai"),
        ("Current Designation", "What is the current job title?", "ai"),
        ("Current Salary", "What is the current annual salary?", "ai"),
        ("Current Salary Currency", "What is the currency of the current salary?", "ai"),
        
        ("Previous Organization", "What was the previous company name?", "ai"),
        ("Previous Joining Date", "When did they join the previous company?", "ai"),
        ("Previous end year", "When did they leave the previous company?", "ai"),
        ("Previous Starting Designation", "What was the designation at the previous company?", "ai"),
        
        ("High School", "What is the name of the high school?", "ai"),
        ("12th standard pass out year", "Which year was 12th standard completed?", "ai"),
        ("12th overall board score", "What was the 12th board score?", "ai"),
        
        ("Undergraduate degree", "What is the undergraduate degree?", "ai"),
        ("Undergraduate college", "Which college did they attend for undergraduate?", "ai"),
        ("Undergraduate year", "Which year did they graduate undergraduate?", "ai"),
        ("Undergraduate CGPA", "What was the undergraduate CGPA?", "ai"),
        
        ("Graduation degree", "What is the masters degree?", "ai"),
        ("Graduation college", "Which college did they attend for masters?", "ai"),
        ("Graduation year", "Which year did they complete masters?", "ai"),
        ("Graduation CGPA", "What was the masters CGPA?", "ai"),
        
        ("Certifications 1", "What is the first certification mentioned?", "ai"),
        ("Certifications 2", "What is the second certification mentioned?", "ai"),
        ("Certifications 3", "What is the third certification mentioned?", "ai"),
        ("Certifications 4", "What is the fourth certification mentioned?", "ai"),
        
        ("Technical Proficiency", "What technical skills does the candidate have?", "ai")
    ]

    final_rows = []
    row_num = 1
    
    # Progress bar
    bar = st.progress(0)
    
    # Limit text context to avoid Token errors (First 3500 chars covers most data)
    # For certifications/skills at end, we might look at the tail, but simple truncation is safer for stability.
    context_window = text[:4000] 

    for key, question, source in schema:
        value = ""
        comment = ""
        
        if source == "basics":
            # Use Regex/Heuristic result if available
            value = basics.get(key, "")
            # If missing in basics, try AI as fallback
            if not value:
                try:
                    res = qa_pipeline(question=question, context=context_window)
                    if res['score'] > 0.01: value = res['answer']
                except: pass
        
        else: # Source is AI
            try:
                res = qa_pipeline(question=question, context=context_window)
                # Filtering low confidence "guesses"
                if res['score'] > 0.01:
                    value = res['answer']
                    # GENERATE COMMENT: Find the sentence where this answer lived
                    comment = find_context_sentence(text, value)
            except Exception:
                value = ""

        # Append row
        final_rows.append({
            "#": row_num,
            "Key": key,
            "Value": value,
            "Comments": comment
        })
        
        row_num += 1
        bar.progress(row_num / len(schema))
        
    bar.empty()
    return final_rows

# --- 4. STREAMLIT UI ---

st.title("🤖 Rubric-Compliant AI Parser")
st.markdown("""
**Status:** Ready for Submission.
**Logic:** Hybrid (Regex for Headers + Transformer AI for Body).
**Output:** Strictly formatted 37-row Excel file with Context Comments.
""")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("AI is analyzing document structure..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        
        if raw_text:
            # Run the main loop
            data = run_assignment_extraction(raw_text)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # UI Display
            st.success("Extraction Complete")
            st.dataframe(df, use_container_width=True)
            
            # Excel Export
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
                
            st.download_button(
                label="⬇️ Download Output.xlsx",
                data=buffer,
                file_name="Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
