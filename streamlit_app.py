import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn.compose._column_transformer as ct

# Backward compatibility fix for unpickling ColumnTransformer across scikit-learn versions
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList:
        pass
    ct._RemainderColsList = _RemainderColsList

# Set page configuration
st.set_page_config(
    page_title="University Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

# App title and description
st.title("🎓 University Cutoff Score Predictor")
st.write("""
Predict university admission cutoff scores based on programme, college campus, 
variant, counseling round, and reservation categories.
""")

# Load the saved preprocessor and model
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_random_forest_model.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
    st.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.exception(e)
    st.stop()

# Create Sidebar / Input form
st.header("Predictor Parameters")

# Hardcoded selections based on unique categories from our training dataset
years = [2024, 2025, 2026]

# User Inputs
year = st.selectbox("Year", years, index=1)

# Simplified dropdowns for categorical columns based on our dataset content
programme = st.selectbox("Programme", [
    "B.A. (Honours) — Economics, History, Political Science, Sociology, Psychology, Geography, Arabic, Bengali, Hindi, Persian, Pali, Sanskrit, Urdu, English, Japanese, Chinese, German, French, AIHC & Archaeology, History of Art, Philosophy, Linguistics, Home Science, Music, Painting, Education, Dance",
    "Shastri (Honours) — Veda, Vyakarana, Sahitya, Jyotish, Dharmashastra, Mimansa, Vedanta, Nyayavaishashika, Prachinavyakarana, Sankhyayoga, Jaindarshan, Bauddhadarshan, Purana, Agamatantra, Dharmagama",
    "B.A. (Honours) LL.B.",
    "B.Com. (Honours) / B.Com. (Honours) Financial Market Management",
    "B.Sc. (Honours) Agriculture"
])

college = st.selectbox("College", [
    "Faculty of Social Sciences",
    "Sanskrit Vidya Dharma Vigyan Sankaya",
    "Mahila Maha Vidyalaya (Maintained by BHU)",
    "Faculty of Arts",
    "D.A.V Post Graduate College"
])

variant = st.text_input("Variant (e.g. general, Paid Fee Course)", value="general")
counseling_round = st.selectbox("Round", ["1", "2", "3", "4", "5", "6", "7"])
category = st.selectbox("Category", ["UR", "EWS", "OBC-NCL", "SC", "ST", "BHU-Ward", "PwBD"])
seat_type = st.selectbox("Seat Type", ["general", "female", "other"])

# Prediction Action
if st.button("Predict Cutoff Score", type="primary"):
    # Assemble input DataFrame matching preprocessor structure
    input_data = pd.DataFrame({
        'year': [year],
        'programme': [programme],
        'college': [college],
        'variant': [variant],
        'round': [counseling_round],
        'category': [category],
        'seat_type': [seat_type]
    })
    
    try:
        # Preprocess and predict
        features_encoded = preprocessor.transform(input_data)
        prediction = model.predict(features_encoded)[0]
        
        st.balloons()
        st.metric(label="Predicted Cutoff Score", value=f"{prediction:.2f}")
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
