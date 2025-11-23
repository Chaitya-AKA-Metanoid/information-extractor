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
    """Loads the QA model. Cached so it doesn't reload on every upload."""
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

# --- 3. PATTERN MATCHING (For Fixed Formats) ---
def extract_contact_info(text):
    """Extracts Email and Phone using Regex (More accurate than AI for these)."""
    contact_data = {}
    
    # Email Regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    contact_data["Email"] = email_match.group(0) if email_match else "Not Found"
    
    # Phone Regex (Supports various formats)
    # Looks for 10 digits, sometimes with +91 or dashes
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    contact_data["Phone"] = phone_match.group(0) if phone_match else "Not Found"
    
    return contact_data

# --- 4. AI EXTRACTION (For Variable Formats) ---
def parse_resume_with_ai(text):
    # The standard fields every recruiter wants
    # Question format: Key -> Natural Language Question
    questions = {
        "Full Name": "What is the candidate's full name?",
        "Current Role": "What is their current or most recent job title?",
        "Current Company": "What is the name of their current or latest company?",
        "Total Experience": "How many years of experience do they have?",
        "Highest Degree": "What is the highest degree or qualification obtained?",
        "University/College": "Which university or college did they attend?",
        "Top Skills": "What are the main technical skills or programming languages listed?",
        "Certifications": "What certifications has the candidate listed?"
    }
    
    data = {}
    
    # Progress bar for UX
    bar = st.progress(0)
    count = 0
    
    for key, question in questions.items():
        try:
            # The AI reads the text and answers the question
            result = qa_model(question=question, context=text)
            
            # Filter low confidence answers (hallucinations)
            if result['score'] > 0.02:
                data[key] = result['answer']
            else:
                data[key] = "" # Leave blank if unsure
                
        except Exception:
            data[key] = ""
            
        count += 1
        bar.progress(count / len(questions))
        
    bar.empty()
    return data

# --- 5. UI & LOGIC ---
st.title("🚀 AI Resume Parser")
st.markdown("Upload a resume (PDF) to extract structured data into Excel/CSV.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading Document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        
    if raw_text:
        # 1. Run Hybrid Extraction
        st.write("🤖 **Analyzing Resume...**")
        
        # A. Pattern Matching (Contact Info)
        contact_info = extract_contact_info(raw_text)
        
        # B. AI Analysis (Semantic Data)
        ai_data = parse_resume_with_ai(raw_text)
        
        # C. Merge Data
        full_profile = {**contact_info, **ai_data}
        
        # 2. Display Result
        df = pd.DataFrame([full_profile]) # Wrap in list to make it a single row
        
        st.success("Parsing Complete!")
        st.dataframe(df, use_container_width=True)
        
        # 3. Download Options
        col1, col2 = st.columns(2)
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        with col1:
            st.download_button(
                "⬇️ Download as CSV",
                csv,
                "resume_data.csv",
                "text/csv",
                key='download-csv'
            )
            
        # Excel Download
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
            
        with col2:
            st.download_button(
                label="⬇️ Download as Excel",
                data=buffer,
                file_name="resume_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
