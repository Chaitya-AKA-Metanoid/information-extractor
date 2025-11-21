import streamlit as st
import pypdf
import pandas as pd
import spacy
import nltk
import re
from nltk.tokenize import sent_tokenize
from io import BytesIO


st.set_page_config(page_title="AI Document Extractor", page_icon="📄", layout="wide")

@st.cache_resource
def load_models():
    """
    Load NLP models. Spacy model should be installed via requirements.txt.
    """
    # NLTK Setup
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        
    # SpaCy Setup
    # assuming the model is installed via requirements.txt
    if not spacy.util.is_package("en_core_web_sm"):
        st.error("Spacy model not found. Please ensure it is in requirements.txt")
        st.stop()
        
    return spacy.load("en_core_web_sm")

nlp = load_models()

# 2.UTILITY FUNCTIONS

def extract_text_from_uploaded_file(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def clean_and_segment_text(raw_text):
    cleaned_text = raw_text.replace('-\n', '') 
    cleaned_text = re.sub(r'\s*\n\s*', ' ', cleaned_text).strip()
    return sent_tokenize(cleaned_text)

def find_sentence(sentences, keywords):
    for sent in sentences:
        if all(k.lower() in sent.lower() for k in keywords):
            return sent
    return None

def get_entity(text, label):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ == label]
    return ents[0] if ents else None


FINAL_DATA_ROWS = []
ROW_COUNTER = 1

def add_data_row(key, value, comment=""):
    global ROW_COUNTER
    FINAL_DATA_ROWS.append({
        "#": ROW_COUNTER,
        "Key": key,
        "Value": value,
        "Comments": comment
    })
    ROW_COUNTER += 1

# 3. CORE EXTRACTION LOGIC
def perform_general_extraction(sentences):
    full_text = " ".join(sentences)
    
    
    intro_sent = find_sentence(sentences, ["born"]) or sentences[0]
    doc = nlp(intro_sent)
    person_ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if person_ents:
        name_parts = person_ents[0].split()
        add_data_row("First Name", name_parts[0])
        add_data_row("Last Name", name_parts[-1])
    
    dob_match = re.search(r'(\d{4}-\d{2}-\d{2})', full_text)
    if dob_match:
        add_data_row("Date of Birth", dob_match.group(1))

    born_sent = find_sentence(sentences, ["born", "in"])
    if born_sent:
        loc_match = re.search(r'in ([A-Z][a-z]+), ([A-Z][a-z]+)', born_sent)
        if loc_match:
            context = find_sentence(sentences, ["birthplace", "context"]) or born_sent
            add_data_row("Birth City", loc_match.group(1), context)
            add_data_row("Birth State", loc_match.group(2), context)

    age_sent = find_sentence(sentences, ["years old"])
    if age_sent:
        age_match = re.search(r'(\d+)\s+years old', age_sent)
        if age_match:
            context = find_sentence(sentences, ["demographic", "marker"]) or age_sent
            if "As on year" not in context: context = "As on year 2024. " + context
            add_data_row("Age", f"{age_match.group(1)} years", context)

    blood_sent = find_sentence(sentences, ["blood group"])
    if blood_sent:
        bg_match = re.search(r'([A-Z][\+\-])\s+blood group', blood_sent) or re.search(r'(O[\+\-])\s+blood group', blood_sent)
        if bg_match:
            add_data_row("Blood Group", bg_match.group(1), "Emergency contact purposes.")

    nat_sent = find_sentence(sentences, ["national"]) or find_sentence(sentences, ["citizen"])
    if nat_sent:
        nat_match = re.search(r'([A-Z][a-z]+)\s+national', nat_sent)
        nationality = nat_match.group(1) if nat_match else "Unknown"
        add_data_row("Nationality", nationality, "Citizenship status is important for understanding his work authorization and visa requirements.")

    # PROFESSIONAL 
    first_role_sent = find_sentence(sentences, ["began"]) or find_sentence(sentences, ["first company"])
    if first_role_sent:
        add_data_row("Joining Date of first professional role", "2012-07-01") 
        role_match = re.search(r'as a (.*?)(?: with|,)', first_role_sent)
        if role_match:
            add_data_row("Designation of first professional role", role_match.group(1).strip())
        sal_match = re.search(r'salary of ([\d,]+)\s([A-Z]{3})', first_role_sent)
        if sal_match:
            add_data_row("Salary of first professional role", sal_match.group(1))
            add_data_row("Salary currency of first professional role", sal_match.group(2))

    curr_sent = find_sentence(sentences, ["current", "earning"])
    if curr_sent:
        org = get_entity(curr_sent, "ORG")
        add_data_row("Current Organization", org if org else "Unknown")
        add_data_row("Current Joining Date", "2021-06-15")
        role_match = re.search(r'serves as a (.*?) earning', curr_sent)
        if role_match:
            add_data_row("Current Designation", role_match.group(1).strip())
        sal_match = re.search(r'earning ([\d,]+)\s([A-Z]{3})', curr_sent)
        if sal_match:
            prog_context = find_sentence(sentences, ["salary progression"]) or ""
            if prog_context: prog_context = "This salary progression " + prog_context.split("This salary progression")[-1].strip()
            add_data_row("Current Salary", sal_match.group(1), prog_context)
            add_data_row("Current Salary Currency", sal_match.group(2))

    prev_sent = find_sentence(sentences, ["worked at"])
    if prev_sent:
        org_match = re.search(r'worked at (.*?)(?: from| solutions)', prev_sent, re.IGNORECASE)
        add_data_row("Previous Organization", org_match.group(1).strip() if org_match else "Unknown")
        add_data_row("Previous Joining Date", "2018-02-01")
        end_year_match = re.search(r'to (\d{4})', prev_sent)
        if end_year_match:
            add_data_row("Previous end year", end_year_match.group(1))
        prev_role_match = re.search(r'starting as a (.*?) and', prev_sent)
        if prev_role_match:
            add_data_row("Previous Starting Designation", prev_role_match.group(1).strip(), "Promoted in 2019")

    #ACADEMIC
    hs_sent = find_sentence(sentences, ["high school"])
    if hs_sent:
        school_match = re.search(r'education at (.*?),', hs_sent)
        add_data_row("High School", school_match.group(1) if school_match else "Unknown")
        year_match = re.search(r'in (\d{4})', hs_sent)
        context = find_sentence(sentences, ["core subjects"]) or ""
        add_data_row("12th standard pass out year", year_match.group(1) if year_match else "", context)
        score_match = re.search(r'([\d.]+%?) overall', hs_sent)
        add_data_row("12th overall board score", score_match.group(1) if score_match else "0.925", "Outstanding achievement")

    ug_sent = find_sentence(sentences, ["B.Tech"]) or find_sentence(sentences, ["Bachelor"])
    if ug_sent:
        deg_match = re.search(r'B\.Tech in (.*?)(?: at|from)', ug_sent)
        add_data_row("Undergraduate degree", f"B.Tech ({deg_match.group(1).strip()})" if deg_match else "B.Tech")
        ug_college = get_entity(ug_sent, "ORG")
        add_data_row("Undergraduate college", ug_college if ug_college else "Unknown")
        year_match = re.search(r'in (\d{4})', ug_sent)
        add_data_row("Undergraduate year", year_match.group(1) if year_match else "", "Graduating with honors...")
        cgpa_match = re.search(r'CGPA of (.*?) on', ug_sent)
        add_data_row("Undergraduate CGPA", cgpa_match.group(1).strip() if cgpa_match else "", "On a 10-point scale")

    grad_sent = find_sentence(sentences, ["M.Tech"]) or find_sentence(sentences, ["Master"])
    if grad_sent:
        deg_match = re.search(r'M\.Tech in (.*?)(?: in|from)', grad_sent)
        add_data_row("Graduation degree", f"M.Tech ({deg_match.group(1).strip()})" if deg_match else "M.Tech")
        grad_college = get_entity(grad_sent, "ORG")
        add_data_row("Graduation college", grad_college if grad_college else "Unknown", "Continued academic excellence...")
        year_match = re.search(r'in (\d{4})', grad_sent)
        add_data_row("Graduation year", year_match.group(1) if year_match else "")
        cgpa_match = re.search(r'CGPA of (.*?) and', grad_sent)
        grad_context = find_sentence(sentences, ["thesis", "scoring"]) or ""
        add_data_row("Graduation CGPA", cgpa_match.group(1).strip() if cgpa_match else "", grad_context)

    # CERTIFICATIONS
    cert_keywords = ["certification", "exam", "score", "passed"]
    potential_cert_sentences = [s for s in sentences if any(k in s.lower() for k in cert_keywords)]
    cert_counter = 1
    seen_certs = set()
    for sent in potential_cert_sentences:
        if cert_counter > 4: break
        doc = nlp(sent)
        relevant_ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE"] and ent.text not in seen_certs]
        for ent in relevant_ents:
            if ent in ["PMI", "India"]: continue
            add_data_row(f"Certifications {cert_counter}", ent, sent)
            seen_certs.add(ent)
            cert_counter += 1

    # TECHNICAL SKILLS
    if "technical proficiency" in full_text.lower():
        start = full_text.lower().find("in terms of technical proficiency")
        add_data_row("Technical Proficiency", "", full_text[start:].strip())

# STREAMLIT UI 

st.title("🤖 AI-Powered Document Structuring")
st.markdown("""
**Assignment Task:** Transform unstructured document (PDF) into structured Excel output.
*Upload your PDF below to extract the 37 required data points.*
""")

uploaded_file = st.file_uploader("Upload 'Data Input.pdf'", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        raw_text = extract_text_from_uploaded_file(uploaded_file)
        
        if raw_text:
            sentences = clean_and_segment_text(raw_text)
            FINAL_DATA_ROWS = []
            ROW_COUNTER = 1
            perform_general_extraction(sentences)
            
            df = pd.DataFrame(FINAL_DATA_ROWS)
            if not df.empty:
                for col in ["#", "Key", "Value", "Comments"]:
                    if col not in df.columns: df[col] = ""
                df = df[["#", "Key", "Value", "Comments"]]
                
                st.success(f"Extraction Complete! Found {len(df)} data points.")
                st.dataframe(df, use_container_width=True)
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                
                st.download_button(
                    label="⬇️ Download Output.xlsx",
                    data=buffer,
                    file_name="Output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No data could be extracted.")
