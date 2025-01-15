import streamlit as st
import pickle
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

# Assume `data` is your DataFrame with a 'combined_features' column
data = pd.read_csv('D:\\client\\project 8\\job_recomd\\updatedjobdata.csv')  # Load your dataset
tfidf = TfidfVectorizer()
tfidf.fit_transform(data['combined_features'])  # Fit the TF-IDF vectorizer

# Save both `data` and `tfidf` into a dictionary
with open('recommendation_engine.pkl', 'wb') as f:
    pickle.dump({'data': data, 'tfidf': tfidf}, f)


# Load the saved model and data
with open('recommendation_engine.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
    data = saved_objects['data']
    tfidf = saved_objects['tfidf']

if data is None or tfidf is None:
    raise ValueError("The pickle file does not contain the required 'data' and 'tfidf' objects.")

# Define recommendation function
def recommend_jobs(user_input, top_n=5):
    tfidf_matrix = tfidf.transform([user_input])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf.transform(data['combined_features']))
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

    

  # Monitoring Workforce Dynamics  

# Preprocess datetime columns
data['publisheddate'] = pd.to_datetime(data['publisheddate'])
data['year'] = data['publisheddate'].dt.year
data['month'] = data['publisheddate'].dt.month

# Streamlit UI
st.title("Monitoring Workforce Dynamics")

# Filter options
st.sidebar.header("Filter Options")
selected_year = st.sidebar.selectbox("Select Year", options=sorted(data['year'].unique()))
selected_month = st.sidebar.selectbox("Select Month", options=sorted(data['month'].unique()))
filtered_data = data[(data['year'] == selected_year) & (data['month'] == selected_month)]

# Key Insights
st.subheader("Key Insights")
st.write(f"Showing data for {selected_year}-{selected_month:02d}")

# 1. Job Categories Trend
st.subheader("Job Categories Trend")
category_counts = filtered_data['job_category'].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax1)
ax1.set_title("Popular Job Categories")
ax1.set_xlabel("Job Category")
ax1.set_ylabel("Number of Jobs")
st.pyplot(fig1)

# 2. Salary Trends
st.subheader("Salary Trends")
fig2, ax2 = plt.subplots()
sns.lineplot(data=filtered_data, x='publisheddate', y='budget', hue='job_category', ax=ax2)
ax2.set_title("Salary Trends Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Budget")
st.pyplot(fig2)

# 3. Job Type Distribution
st.subheader("Job Type Distribution")
job_type_counts = filtered_data['RemoteWork'].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(job_type_counts, labels=job_type_counts.index, autopct='%1.1f%%', startangle=140)
ax3.set_title("Remote vs. On-Site Jobs")
st.pyplot(fig3)

# 4. Summary Table
st.subheader("Summary Table")
st.dataframe(filtered_data[['title', 'job_category', 'country', 'budget', 'RemoteWork']])