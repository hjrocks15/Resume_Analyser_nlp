import streamlit as st
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract text from PD
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Extract top N keywords using TF-IDF
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform([text])
    keywords = sorted(
        zip(vectorizer.get_feature_names_out(), X.toarray()[0]),
        key=lambda x: x[1],
        reverse=True
    )
    return [kw[0] for kw in keywords[:top_n]]

# Compute cosine similarity between resume and JD
def compute_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Find missing keywords
def find_missing_keywords(job_keywords, resume_text):
    missing = [kw for kw in job_keywords if kw not in resume_text]
    return missing

# Streamlit UI
st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("Resume vs Job Description Matcher")
st.markdown("Upload your resume and paste the job description to see your match score!")

resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Text")

if resume_file and job_description:
    with st.spinner("Analyzing..."):
        resume_raw = extract_text_from_pdf(resume_file)
        resume_clean = preprocess_text(resume_raw)
        jd_clean = preprocess_text(job_description)

        job_keywords = extract_keywords(jd_clean, top_n=15)
        match_score = compute_similarity(resume_clean, jd_clean)
        missing_keywords = find_missing_keywords(job_keywords, resume_clean)

    st.success("Analysis Complete!")
    st.metric("🔍 Job Fit Score", f"{match_score * 100:.2f} %")

    st.subheader("Top Keywords in Job Description")
    st.write(", ".join(job_keywords))

    st.subheader("Missing Keywords in Your Resume")
    if missing_keywords:
        st.write(", ".join(missing_keywords))
    else:
        st.write("Your resume covers all the main keywords")

else:
    st.info("Please upload a resume and paste a job description to proceed.")
