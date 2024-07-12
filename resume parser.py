import PyPDF2
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import os

# Download NLTK data (for first-time use)
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

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

def extract_section(text, section_name):
    pattern = re.compile(rf'({section_name})(.*?)(Education|Experience|Skills|Projects|References|$)', re.S)
    section = pattern.search(text)
    if section:
        return section.group(2).strip()
    return ''

def parse_resume(text):
    contact_info = extract_contact_info(text)
    education = extract_section(text, 'Education')
    experience = extract_section(text, 'Experience')
    skills = extract_section(text, 'Skills')
    
    resume_data = {
        'Phone': ', '.join(contact_info[0]) if contact_info[0] else None,
        'Email': ', '.join(contact_info[1]) if contact_info[1] else None,
        'Education': education,
        'Experience': experience,
        'Skills': skills
    }
    
    return resume_data

def save_to_csv(data, output_file):
    # Convert to DataFrame
    df = pd.DataFrame([data])
    df.to_csv(output_file, index=False)

def main():
    file_path = input("Please enter the path to the resume PDF file: ")
    
    try:
        text = extract_text_from_pdf(file_path)
        cleaned_text = preprocess_text(text)
        resume_data = parse_resume(cleaned_text)
        
        # Generate the output file path
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_file = f"{name}_parsed.csv"
        
        print("Parsed Resume Data:")
        print(resume_data)
        
        save_to_csv(resume_data, output_file)
        print(f"Data has been saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as ve:
        print(ve)

if __name__ == "__main__":
    main()
