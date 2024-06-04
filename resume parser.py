import PyPDF2
import docx
import re
import json
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Download NLTK data (for first-time use)
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError('Unsupported file format')

def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def extract_contact_info(text):
    phone_pattern = re.compile(r'\(?\b[0-9]{3}[-.\s)]{0,2}[0-9]{3}[-.\s]{0,1}[0-9]{4}\b')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_matches = phone_pattern.findall(text)
    email_matches = email_pattern.findall(text)
    return phone_matches, email_matches

def extract_education(text):
    education_section = re.search(r'(Education|Academic Background|Educational Qualifications|Academic Qualifications)(.*?)(Experience|Skills|Projects|References|$)', text, re.S)
    if education_section:
        education_text = education_section.group(2)
        return education_text.strip()
    return ''

def extract_experience(text):
    experience_section = re.search(r'(Experience|Work History|Professional Experience|Employment History)(.*?)(Education|Skills|Projects|References|$)', text, re.S)
    if experience_section:
        experience_text = experience_section.group(2)
        return experience_text.strip()
    return ''

def extract_skills(text):
    skills_section = re.search(r'(Skills|Technical Skills|Proficiencies)(.*?)(Experience|Education|Projects|References|$)', text, re.S)
    if skills_section:
        skills_text = skills_section.group(2)
        return skills_text.strip()
    return ''

def parse_resume(text):
    contact_info = extract_contact_info(text)
    education = extract_education(text)
    experience = extract_experience(text)
    skills = extract_skills(text)
    
    resume_data = {
        'contact_info': {
            'phone': contact_info[0] if contact_info[0] else None,
            'email': contact_info[1] if contact_info[1] else None
        },
        'education': education,
        'experience': experience,
        'skills': skills
    }
    
    return resume_data

def main():
    file_path = input("Please enter the path to the resume file (PDF or DOCX): ")
    
    try:
        text = extract_text(file_path)
        cleaned_text = preprocess_text(text)
        resume_data = parse_resume(cleaned_text)
        
        print("Parsed Resume Data:")
        print(json.dumps(resume_data, indent=2))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as ve:
        print(ve)

if __name__ == "__main__":
    main()
