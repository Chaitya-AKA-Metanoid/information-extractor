import streamlit as st
import pypdf
import pandas as pd
import spacy
import re
from io import BytesIO
from collections import Counter

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Free-Form Resume Parser", page_icon="🔓", layout="wide")

@st.cache_resource
def load_nlp():
    # Load the small English model for speed and efficiency
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# --- 2. SKILLS DATABASE (The "Knowledge Base") ---
# A simple list of common keywords to look for. 
# In a real app, this would be a massive external file or database.
SKILLS_DB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "react", "angular", "vue",
    "node.js", "django", "flask", "fastapi", "sql", "mysql", "postgresql", "mongodb", "redis",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
    "tableau", "power bi", "excel", "jira", "agile", "scrum", "linux", "bash", "marketing", "seo",
    "communication", "management", "leadership", "project management", "sales", "analysis"
]

# --- 3. UTILITY FUNCTIONS ---

def extract_text(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except:
        return None

def extract_emails(text):
    # Standard regex for emails
    return list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))

def extract_phones(text):
    # Regex for finding phone-like patterns (10+ digits)
    # Filters out years (like 2020-2021) by checking length and surrounding chars
    matches = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    return list(set([m for m in matches if len(re.sub(r'\D', '', m)) >= 10]))

def extract_skills(text):
    # Simple keyword matching against our DB
    found_skills = []
    tokens = [t.lower() for t in text.split()] # Simple tokenization
    # Also check bi-grams for things like "machine learning"
    text_lower = text.lower()
    
    for skill in SKILLS_DB:
        if skill in text_lower:
            found_skills.append(skill)
            
    return list(set(found_skills))

# --- 4. CORE PARSING ENGINE ---

def parse_resume(text):
    doc = nlp(text)
    data = []

    # 1. Contact Info (High Precision Regex)
    emails = extract_emails(text)
    phones = extract_phones(text)
    
    if emails: data.append({"Category": "Contact", "Key": "Email", "Value": ", ".join(emails)})
    if phones: data.append({"Category": "Contact", "Key": "Phone", "Value": ", ".join(phones)})

    # 2. NLP Entities (Organizations, People, Locations)
    # We count frequencies to find the most prominent ones (likely the candidate's name or top company)
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    gpe = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"] # Geo-Political Entities (Cities/Countries)
    dates = [ent.text.strip() for ent in doc.ents if ent.label_ == "DATE"]
    persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

    # Heuristic: The first PERSON found is usually the candidate
    if persons:
        data.insert(0, {"Category": "Profile", "Key": "Candidate Name", "Value": persons[0]})

    # Add Lists of extracted entities
    if orgs:
        # Filter duplicates and join
        unique_orgs = list(set(orgs))
        data.append({"Category": "Experience/Education", "Key": "Organizations Found", "Value": ", ".join(unique_orgs)})
    
    if gpe:
        unique_gpe = list(set(gpe))
        data.append({"Category": "Locations", "Key": "Places Mentioned", "Value": ", ".join(unique_gpe)})

    if dates:
        unique_dates = list(set(dates))
        data.append({"Category": "Timeline", "Key": "Dates Mentioned", "Value": ", ".join(unique_dates)})

    # 3. Skills Extraction (Keyword Matching)
    skills = extract_skills(text)
    if skills:
        data.append({"Category": "Skills", "Key": "Tech Stack", "Value": ", ".join(skills)})
        
    return data

# --- 5. STREAMLIT UI ---

st.title("🔓 Free-Form Resume Parser")
st.markdown("No rules. No fixed rows. Just extracts everything it finds.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Parsing..."):
        raw_text = extract_text(uploaded_file)
        
        if raw_text:
            # Run extraction
            extracted_data = parse_resume(raw_text)
            
            # Convert to DataFrame
            df = pd.DataFrame(extracted_data)
            
            st.success(f"Found {len(extracted_data)} data points!")
            
            # Display
            st.dataframe(df, use_container_width=True)
            
            # Download
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
                
            st.download_button(
                label="⬇️ Download Extraction",
                data=buffer,
                file_name="extracted_resume_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
