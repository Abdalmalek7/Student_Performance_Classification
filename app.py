import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ---------------------------------
# Load Model
# ---------------------------------
model = joblib.load("exam_score_calss_model.pkl")

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(
    page_title="Student Performance Classification",
    layout="wide"
)

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Feature Explanation", "Prediction"]
)

# ---------------------------------
# PAGE 1: Project Overview
# ---------------------------------
if page == "Project Overview":

    st.title("🎓 Student Performance Classification")

    image = Image.open("dom-fou-YRMWVcdyhmI-unsplash.jpg")
    st.image(image, use_container_width=True)

    st.markdown("## 📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Rows", "≈ 82,000")
    with col2:
        
        st.metric("Number of Features", "31")
    st.markdown("""
### 🔍 What is this dataset about?
This dataset is designed to analyze and predict students' academic performance by 
capturing a wide range of behavioral, psychological, and lifestyle factors.

It includes information related to:
- **Student behavior** (study habits, time management, screen usage)
- **Mental health** (stress level, exam anxiety, psychological well-being)
- **Family support** (parental support and family background)
- **Lifestyle patterns** (sleep quality, diet, and daily routines)

---

### 🎯 Project Objective
The main goal of this project is to **predict the student's final exam score (Exam Score)**  
based on these combined factors, helping to understand what truly drives academic success.

---

### 🛠 What did we do?
- Data Cleaning and Validation  
- Feature Engineering (e.g., study efficiency, sleep quality, screen time impact)  
- Encoding Categorical Features  
- Scaling Numerical Features  
- Training and Comparing Multiple Regression Models  
- Hyperparameter Tuning using **Randomized Search**  
- Selecting the Best-Performing Model  

---

### 💡 Why is this useful?
- **Early identification** of students at risk of low academic performance  
- Helping **schools and educators** make data-driven interventions  
- Supporting **parents** in understanding factors affecting their children’s performance  
- Guiding **students** toward improving specific behaviors to enhance outcomes  
- Enabling **data-driven academic decision-making** instead of intuition-based judgment  

This project demonstrates how data can be transformed into actionable insights  
to improve educational outcomes.
""")


# ---------------------------------
# PAGE 2: Feature Explanation
# ---------------------------------
elif page == "Feature Explanation":

    st.title("📘 Feature Description & Input Guide")

    st.markdown(
        """
        This page explains each feature used in the model and how it impacts
        the student's academic performance prediction.
        """
    )

    feature_info = {
        "sleep_hours": 
        "Average number of hours the student sleeps per day. Adequate sleep improves focus, memory, and academic performance.",

        "exercise_frequency": 
        "Number of times the student exercises per week. Regular exercise is linked to better mental health and concentration.",

        "stress_level": 
        "Measures the student's stress level on a scale from 1 to 10. Higher stress often negatively impacts academic outcomes.",

        "screen_time": 
        "Total daily screen time in hours. Excessive screen usage may reduce study time and sleep quality.",

        "study_environment": 
        "The environment where the student usually studies (e.g., quiet room, library). A better environment enhances productivity.",

        "access_to_tutoring": 
        "Indicates whether the student has access to additional tutoring or academic support.",

        "motivation_level": 
        "Represents how motivated the student is to study, rated from 1 to 10. Higher motivation usually leads to better performance.",

        "exam_anxiety_score": 
        "Measures anxiety level before exams. High anxiety can reduce exam performance despite good preparation.",

        "study_efficiency": 
        "Represents how effectively the student studies within a given time. Higher values indicate better focus and learning quality.",

        "screen_time_penalty": 
        "Quantifies the negative impact of excessive screen time on academic performance."
    }

    df_features = pd.DataFrame(
        feature_info.items(),
        columns=["Feature Name", "Description"]
    )

    st.dataframe(df_features, use_container_width=True)

# ---------------------------------
# PAGE 3: Prediction
# ---------------------------------
elif page == "Prediction":

    st.title("🤖 Student Performance Classification")
    st.markdown("### Enter Student Information")

    col1, col2 = st.columns(2)

    with col1:
        sleep_hours = st.slider(
            "Sleep Hours per Day",
            min_value=4.0,
            max_value=12.0,
            value=7.0,
            step=0.1
        )

        exercise_frequency = st.slider(
            "Exercise Frequency (per week)",
            min_value=0,
            max_value=7,
            value=3
        )

        stress_level = st.slider(
            "Stress Level",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        screen_time = st.slider(
            "Daily Screen Time (hours)",
            min_value=0.3,
            max_value=21.0,
            value=5.0,
            step=0.1
        )

        motivation_level = st.slider(
            "Motivation Level",
            min_value=1,
            max_value=10,
            value=6
        )

    with col2:
        exam_anxiety_score = st.slider(
            "Exam Anxiety Score",
            min_value=5.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )

        study_efficiency = st.number_input(
            "Study Efficiency",
            min_value=0.0,
            max_value=5.75,
            value=2.0,
            step=0.05
        )

        screen_time_penalty = st.number_input(
            "Screen Time Penalty",
            min_value=0.0,
            max_value=90.0,
            value=10.0,
            step=1.0
        )

        study_environment = st.selectbox(
            "Study Environment",
            ['Quiet Room', 'Library', 'Co-Learning Group', 'Dorm', 'Cafe']
        )

        access_to_tutoring = st.selectbox(
            "Access to Tutoring",
            ["Yes", "No"]
        )

    if st.button("🎯 Predict Performance"):

        input_data = pd.DataFrame([{
            'sleep_hours': sleep_hours,
            'exercise_frequency': exercise_frequency,
            'stress_level': stress_level,
            'screen_time': screen_time,
            'study_environment': study_environment,
            'access_to_tutoring': access_to_tutoring,
            'motivation_level': motivation_level,
            'exam_anxiety_score': exam_anxiety_score,
            'study_efficiency': study_efficiency,
            'screen_time_penalty': screen_time_penalty
        }])
        class_mapping = {
            0: "❌ At Risk",
            1: "⚠️ Average",
            2: "✅ High Performer"
        }
        prediction = model.predict(input_data)[0]
        prediction_label = class_mapping.get(prediction, "Unknown")

        st.success(f"📊 Predicted Performance Level: **{prediction_label}**")

        
        



