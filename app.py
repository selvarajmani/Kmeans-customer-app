import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("cleaned_customer_data.csv")

# Select features
X = df[['Annual_income', 'Spending_score']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Define custom cluster labels
labels = {
    0: "Mid-Range Active",
    1: "Upsell Candidate",
    2: "Top Customer",
    3: "Low Engagement",
    4: "Luxury Dormant"
}

st.title("🧠 Customer Segmentation Predictor")

st.write("Enter a customer's Annual Income and Spending Score to predict their segment.")

income = st.number_input("Enter Annual Income (in $K):", min_value=0.0, step=1.0)
score = st.number_input("Enter Spending Score (1–100):", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict Segment"):
    user_input = scaler.transform([[income, score]])
    prediction = kmeans.predict(user_input)[0]
    st.success(f"🎯 Predicted Cluster: {prediction} - {labels[prediction]}")