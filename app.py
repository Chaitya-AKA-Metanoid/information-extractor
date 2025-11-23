import streamlit as st
import pypdf
import pandas as pd
import re
from transformers import pipeline
from io import BytesIO

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="Universal Resume Parser", page_icon="👔", layout="wide")

@st.cache_resource
def load_model():
    # Using a slightly more capable model for document reading
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_model()

# --- 2. TEXT EXTRACTION ---
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

# --- 3. HYBRID EXTRACTION (Regex + AI) ---

def extract_header_info(text):
    """
    Extracts Name, Email, Phone using Regex and heuristic position.
    Reliable because contact info is almost always at the top.
    """
    data = {}
    
    # 1. NAME HEURISTIC
    # Assume the first non-empty line that isn't a label (like "Resume") is the Name.
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Simple blocklist of words that might appear at top but aren't names
    blocklist = ["RESUME", "CURRICULUM VITAE", "CV", "PROFILE", "CONTACT"]
    
    for line in lines[:5]: # Check first 5 lines
        if len(line.split()) <= 4 and line.upper() not in blocklist:
            # Validate it looks like a name (Title Case, no numbers)
            if not any(char.isdigit() for char in line):
                data["Full Name"] = line
                name_parts = line.split()
                data["First Name"] = name_parts[0]
                data["Last Name"] = name_parts[-1] if len(name_parts) > 1 else ""
                break
    
    if "Full Name" not in data:
        data["Full Name"] = "Not Found"

    # 2. EMAIL REGEX
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    data["Email"] = email_match.group(0) if email_match else "Not Found"
    
    # 3. PHONE REGEX
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    data["Phone"] = phone_match.group(0) if phone_match else "Not Found"
    
    return data

def extract_details_with_ai(text):
    """
    Uses AI to find complex fields like Experience, Degree, and Skills.
    Truncates text to fit model if needed (or you can implement chunking).
    """
    # Truncate text for the model (it can handle ~512 tokens, approx 3000 chars)
    # We focus on the first 3000 characters for Summary/Experience/Education
    # and the last 1000 characters for Skills/Certifications if needed.
    # For simplicity here, we take the first 4000 chars which covers most 1-2 page resumes.
    model_text = text[:4000] 
    
    questions = {
        "Highest Degree": "What is the highest degree or qualification?",
        "University": "Which university or college did they attend?",
        "Total Experience": "How many years of total work experience?",
        "Latest Job Title": "What is their current or most recent job designation?",
        "Latest Company": "What is the name of their current or recent company?",
        "Key Skills": "What are the main technical skills listed?",
        "Certifications": "What certifications are mentioned?"
    }
    
    results = {}
    
    for key, question in questions.items():
        try:
            response = qa_model(question=question, context=model_text)
            
            # Threshold filtering
            if response['score'] > 0.01:
                results[key] = response['answer']
            else:
                results[key] = "Not Found"
        except:
            results[key] = "Error"
            
    return results

# --- 4. UI & LOGIC ---
st.title("🚀 Hybrid AI Resume Parser")
st.markdown("Extracts Name/Contact via Regex (High Precision) and Details via AI (High Recall).")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading Document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        
    if raw_text:
        st.write("✅ **Document successfully read. Extracting data...**")
        
        # 1. Get Header Info (The things AI missed)
        header_data = extract_header_info(raw_text)
        
        # 2. Get Body Info (The things AI is good at)
        body_data = extract_details_with_ai(raw_text)
        
        # 3. Combine
        full_profile = {**header_data, **body_data}
        
        # 4. Display
        df = pd.DataFrame([full_profile])
        st.dataframe(df, use_container_width=True)
        
        # 5. Download
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
            
        st.download_button(
            label="⬇️ Download as Excel",
            data=buffer,
            file_name="parsed_resume.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
