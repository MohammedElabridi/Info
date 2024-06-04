import streamlit as st
import pandas as pd
import PyPDF2
import os
import csv
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import spacy
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from docx import Document

# Load the Spacy English model
nlp = spacy.load('en_core_web_sm')

# Read skills from CSV file
skills_csv_path = r'/Users/elabridi/Desktop/Job-Recommendation-System/src/data/skills.csv'  # Update the path to the correct CSV file location
with open(skills_csv_path, 'r') as file:
    csv_reader = csv.reader(file)
    skills_list = [row for row in csv_reader]

# Create pattern dictionaries from skills
skill_patterns = [[{'LOWER': skill}] for skill in skills_list[0]]

# Create a Matcher object
matcher = Matcher(nlp.vocab)

# Add skill patterns to the matcher
for pattern in skill_patterns:
    matcher.add('Skills', [pattern])

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def skills_extractor(file_path):
    # Extract text from PDF
    resume_text = extract_text_from_pdf(file_path)
    # Extract skills from resume text
    skills = list(extract_skills(resume_text))
    return skills

# Load dataset
jd_df = pd.read_csv(r'/Users/elabridi/Downloads/UpdatedResumeDataSet.csv')  # Update the path to the correct CSV file location

# Function to generate n-grams
def ngrams(string, n=3):
    string = fix_text(string)  # fix text
    string = string.encode("ascii", errors="ignore").decode()  # remove non-ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single
    string = ' ' + string + ' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# Initialize the vectorizer and NearestNeighbors model
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
nbrs = None

# Function to get nearest neighbors
def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

# Streamlit app
def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

    if uploaded_file is not None:
        # Process resume and recommend jobs
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = "temp_resume.pdf"
        
        # Extract resume skills
        resume_skills = skills_extractor(file_path)
        skills = [' '.join(word for word in resume_skills)]
        
        # Feature Engineering
        global nbrs
        tfidf = vectorizer.fit_transform(skills)
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        
        jd_descriptions = jd_df['Resume'].astype('U').tolist()
        distances, indices = getNearestN(jd_descriptions)

        matches = []
        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)
            temp = [dist]
            matches.append(temp)
        
        matches = pd.DataFrame(matches, columns=['Match confidence'])
        jd_df['match'] = matches['Match confidence']
        
        # Recommend Top 5 Jobs based on candidate resume
        recommended_jobs = jd_df.sort_values('match').head(1)

        # Extract only the 'Category' column of the recommended jobs
        categories = recommended_jobs['Category'].tolist()

        # Display recommended job categories as plain text
        st.title("Recommended Job Categories:")
        for category in categories:
            st.write(category)

# Run the Streamlit app
if __name__ == '__main__':
    main()
