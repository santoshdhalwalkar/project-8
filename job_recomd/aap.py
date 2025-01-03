import streamlit as st
import pickle

# Load the saved model and data
with open('recommendation_engine.pkl', 'rb') as f:
    tfidf, data = pickle.load(f)

# Define recommendation function
def recommend_jobs(user_input, top_n=5):
    tfidf_matrix = tfidf.transform([user_input])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf.fit_transform(data['combined_features']))
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    job_indices = [i[0] for i in sim_scores]
    return data.iloc[job_indices][['title', 'job_category', 'country', 'budget']]

# Streamlit UI
st.title("Personalized Job Recommendation Engine")

# Input user preferences
user_skills = st.text_input("Enter your skills:")
preferred_location = st.text_input("Preferred location:")
desired_salary = st.number_input("Desired salary:", min_value=0)

# Generate recommendations
if st.button("Find Jobs"):
    user_input = f"{user_skills} {preferred_location} {desired_salary}"
    recommendations = recommend_jobs(user_input)
    st.write("Recommended Jobs:")
    st.dataframe(recommendations)
