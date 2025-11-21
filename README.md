# information-extractor

# AI-Powered Document Structuring & Data Extraction

## Overview
This project is an AI-backed solution designed to transform unstructured PDF documents (specifically candidate profiles) into structured Excel formats. It leverages Natural Language Processing (NLP) to ensure 100% data capture, converting unstructured text into precise Key:Value pairs with contextual comments.

**Live Demo:** ['https://information-extractor-fgvhwyvfypm3k7nsxinypr.streamlit.app/']

## 🚀 Features
* **Dynamic Extraction:** Uses generic keyword searching and NER (Named Entity Recognition) rather than fixed sentence indexing, making it robust to layout changes.
* **Context Preservation:** Captures full sentences and logical context for specific data points in a dedicated "Comments" column.
* **As-Is Data Fidelity:** Retains original formatting for salaries, dates, and scores as per assignment requirements.
* **Tech Stack:** Python 3.11, Streamlit, SpaCy, NLTK, PyPDF, Pandas.

## 📂 Project Structure
* `app.py`: Main application script containing the Streamlit UI and extraction logic.
* `requirements.txt`: List of Python dependencies (including pinned Numpy version for compatibility).
* `README.md`: Project documentation.

## 🧩 Algorithm Logic
1.  **Text Segmentation:** Uses NLTK to break raw PDF text into logical sentences.
2.  **Entity Detection:** Uses SpaCy to identify Organizations (ORG) and People (PERSON).
3.  **Keyword Search:** Scans for semantic triggers (e.g., "born", "salary", "B.Tech") to locate relevant information regardless of its position in the document.
4.  **Data Structuring:** Maps extracted data to the specific 37-row schema required by the assignment.
