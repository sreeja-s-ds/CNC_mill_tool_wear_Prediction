import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
import pickle
from tensorflow.keras.models import load_model

# Streamlit Page Configuration
st.set_page_config(page_title="CNC Tool Wear Prediction", layout="wide")

# Background Image from Local
def add_bg_from_local(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Please check the file path.")

# ✅ Replace with your actual local path
add_bg_from_local(r"D:\00CNC\cncbg.jpeg")

# Load Models and Scalers (with compile=False to avoid deserialization errors)
@st.cache_resource
def load_tool_condition_model():
    return load_model('tool_condition_model.h5', compile=False)

@st.cache_resource
def load_multi_output_model():
    return load_model('multi_output_model.h5', compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.joblib')

# Load all resources
tool_condition_model = load_tool_condition_model()
multi_output_model = load_multi_output_model()
scaler = load_scaler()

expected_features = scaler.feature_names_in_

# Preprocessing Function
def preprocess_input(file):
    try:
        df = pd.read_csv(file)
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_features]

        X_scaled = scaler.transform(df)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        return X_reshaped, df
    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
        return None, None

# Prediction Functions
def predict_tool_condition(X):
    pred = tool_condition_model.predict(X)
    labels = np.argmax(pred, axis=1)
    
    # Hardcoded mapping: adjust if your model's label order is different
    decoded = ['Unworn' if label == 0 else 'Worn' for label in labels]
    return decoded


def predict_multi_outputs(X):
    preds = multi_output_model.predict(X)
    machining_finalized = (preds[0] > 0.5).astype(int).flatten()
    passed_visual = (preds[1] > 0.5).astype(int).flatten()
    return machining_finalized, passed_visual

# Streamlit App UI
st.title("🔧 CNC Tool Wear & Quality Prediction App")

st.markdown("""
### 📋 About the Project
This application predicts:
- 🛠 **Tool Condition** (Unworn / Worn)
- ⚙️ **Machining Finalized** (Yes/No)
- 🔍 **Passed Visual Inspection** (Yes/No)

Upload sensor data (CSV) to get predictions based on **Deep Learning LSTM models**.
""")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=['csv'])

if uploaded_file is not None:
     # ✅ Show what features your scaler expects
    expected_features = scaler.feature_names_in_
    st.write("✅ Expected Features:", expected_features)
    st.write("✅ Number of Expected Features:", len(expected_features))

    # ✅ Show what columns your uploaded file has
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded CSV Columns:", df.columns.tolist())
    st.write("📊 Number of Uploaded CSV Columns:", len(df.columns))

    # Preprocess and continue as usual
    X_input, original_df = preprocess_input(uploaded_file)
    

    if X_input is not None:
        tool_condition_preds = predict_tool_condition(X_input)
        machining_preds, visual_preds = predict_multi_outputs(X_input)

        machining_labels = ['No' if val == 0 else 'Yes' for val in machining_preds]
        visual_labels = ['No' if val == 0 else 'Yes' for val in visual_preds]

        result_df = original_df.copy()
        result_df['Tool Condition'] = tool_condition_preds
        result_df['Machining Finalized'] = machining_labels
        result_df['Passed Visual Inspection'] = visual_labels

        st.success("✅ Predictions Completed!")

        st.write("### 🔍 Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv_output, file_name="cnc_predictions.csv", mime="text/csv")
    else:
        st.error("Please upload a valid CSV file.")
