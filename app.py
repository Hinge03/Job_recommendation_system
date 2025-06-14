from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load saved models and data
with open('tfidf_recruiter.pkl', 'rb') as f:
    tfidf_recruiter = pickle.load(f)

with open('tfidf_candidate.pkl', 'rb') as f:
    tfidf_candidate = pickle.load(f)

df = pd.read_pickle('candidate_profiles.pkl')   # recruiter sees candidate list
df1 = pd.read_pickle('job_postings.pkl')        # candidate sees job list

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Recruiter route
@app.route('/recruiter', methods=['POST'])
def recruiter():
    skills = request.form['skills']
    experience = request.form.get('experience', '')

    cleaned_skills = clean_text(skills).split()

    skill_mask = df['Skills'].apply(
        lambda x: all(skill in clean_text(x) for skill in cleaned_skills)
    )
    filtered_jobs = df[skill_mask]

    if experience:
        cleaned_exp = clean_text(experience)
        exp_mask = filtered_jobs['Experience'].apply(
            lambda x: cleaned_exp in clean_text(x)
        )
        filtered_jobs = filtered_jobs[exp_mask]

    if len(filtered_jobs) == 0:
        return render_template("result.html", result="No matching candidates found with these criteria")

    user_query = clean_text(skills)
    if experience:
        user_query += ' ' + clean_text(experience)

    query_vec = tfidf_recruiter.transform([user_query])
    filtered_matrix = tfidf_recruiter.transform(filtered_jobs['combined_features'])
    cosine_sim = linear_kernel(query_vec, filtered_matrix).flatten()

    sim_indices = cosine_sim.argsort()[-10:][::-1]
    result_df = filtered_jobs.iloc[sim_indices][['Name', 'Job_title', 'Experience', 'Skills', 'Company', 'Work_Location']]

    return render_template("result.html", tables=[result_df.to_html(classes='data', header="true", index=False)], title="Result")

# Candidate route
@app.route('/candidate', methods=['POST'])
def candidate():
    skills = request.form['skills']
    cleaned_skills = clean_text(skills).split()

    skill_mask = df1['Skills'].apply(
        lambda x: all(skill in clean_text(x) for skill in cleaned_skills)
    )
    filtered_jobs = df1[skill_mask]

    if len(filtered_jobs) == 0:
        return render_template("result.html", result="No matching jobs found with these skills")

    user_query = clean_text(skills)
    query_vec = tfidf_recruiter.transform([user_query])  # recruiter vector used for both
    filtered_matrix = tfidf_recruiter.transform(filtered_jobs['combined_features'])
    cosine_sim = linear_kernel(query_vec, filtered_matrix).flatten()

    sim_indices = cosine_sim.argsort()[-10:][::-1]
    result_df = filtered_jobs.iloc[sim_indices][['Job_title', 'Location', 'Company', 'Required_experience', 'Skills']]

    return render_template("result.html", tables=[result_df.to_html(classes='data', header="true", index=False)], title="Results")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)