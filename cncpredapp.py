import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from tensorflow.keras.models import load_model
from fpdf import FPDF

st.set_page_config(page_title="Tool Wear Prediction", page_icon="⚙️🛠", layout="wide")

# Set background
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("cncbg.jpeg")

@st.cache_resource
def load_models():
    tool_model = load_model("tool_condition_model.h5")
    multi_model = load_model("multi_output_model.h5")
    scaler = joblib.load("scaler.joblib")
    return tool_model, multi_model, scaler

tool_condition_model, multi_output_model, scaler = load_models()

top_feature_cols = [
    'No', 'Y1_OutputCurrent', 'clamp_pressure',
    'M1_CURRENT_FEEDRATE', 'X1_OutputCurrent', 'feedrate', 'X1_CommandPosition',
    'X1_ActualPosition', 'Y1_CommandPosition', 'X1_OutputVoltage', 'Y1_OutputVoltage',
    'Z1_CommandPosition', 'S1_OutputCurrent', 'X1_CurrentFeedback',
    'M1_CURRENT_PROGRAM_NUMBER', 'Y1_ActualVelocity', 'Y1_CommandVelocity', 'S1_ActualAcceleration'
]

user_input_features = [f for f in top_feature_cols if f != 'No']
sequence_length = 10

feature_guidance = {
    'clamp_pressure': {'Worn': '3–4', 'Unworn': '7–9'},
    'feedrate': {'Worn': '3–10', 'Unworn': '90–140'},
    'M1_CURRENT_FEEDRATE': {'Worn': '3–20', 'Unworn': '90–140'},
    'Y1_OutputCurrent': {'Worn': '320–330', 'Unworn': '90–150'},
}

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #0f172a, #1d4ed8, #06b6d4);
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

menu = ["Home", "Predict"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🔨 CNC Tool Wear Prediction")
    st.markdown("""
    <div style='color: green;'>
        <h2>About this App 🛠</h2>
        <p>Predict <b>Tool Condition (Worn/Unworn)</b>, <b>Machining Finalization</b>, and <b>Visual Inspection</b> using CNC data.</p>
    </div>
                
     ## 🌍 How is this app Useful?
        - **Reduces Downtime:** Predicting tool wear in advance helps preventing unexpected failures.
        - **Cost Savings:** Avoids unnecessary tool replacements and optimizes machining efficiency.
        - **Improved Quality:** Ensures better machining results by maintaining tool health.
                
        📌 This tool empowers industries by providing insights about tool wear, thereby improving productivity and reducing maintenance costs.
        
    """, unsafe_allow_html=True)

elif choice == "Predict":
    st.title("🎛 Tool Wear Prediction Panel ⚙️")

    st.sidebar.header("📊 Feature Input & Guidance")
    with st.sidebar.expander("ℹ️ Feature Ranges (Worn vs. Unworn)"):
        for key, val in feature_guidance.items():
            st.write(f"**{key}** — Worn: {val['Worn']} | Unworn: {val['Unworn']}")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    input_df = None

    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            missing_cols = [col for col in top_feature_cols if col not in df_uploaded.columns]
            if missing_cols:
                st.error(f"❗ Uploaded CSV is missing columns: {', '.join(missing_cols)}")
            else:
                if 'No' not in df_uploaded.columns:
                    df_uploaded['No'] = 0.0
                input_df = df_uploaded[top_feature_cols].iloc[:sequence_length]
                if len(input_df) < sequence_length:
                    padding = pd.DataFrame([input_df.iloc[-1]] * (sequence_length - len(input_df)))
                    input_df = pd.concat([input_df, padding], ignore_index=True)
        except Exception as e:
            st.error(f"❗ Error reading file: {e}")

    else:
        user_data = {f: st.sidebar.number_input(f, value=0.0) for f in user_input_features}
        user_data['No'] = 0.0
        input_df = pd.DataFrame([user_data] * sequence_length)

    if st.sidebar.button("Predict") and input_df is not None:

        if (input_df['clamp_pressure'] < 6).any():
            st.error("⚠️ Clamping Pressure too low! Risk of defective parts.")

        scaled_input = scaler.transform(input_df[top_feature_cols])
        sequence_input = np.array(scaled_input).reshape(1, sequence_length, len(top_feature_cols))

        tool_probs = tool_condition_model.predict(sequence_input, verbose=0)
        prob_worn = tool_probs[0][0]

        threshold = 0.4
        tool_class = int(prob_worn > threshold)
        tool_label = 'Worn' if tool_class == 1 else 'Unworn'
        tool_conf = prob_worn * 100 if tool_class == 1 else (1 - prob_worn) * 100

        multi_preds = multi_output_model.predict(sequence_input, verbose=0)
        machining_prob = multi_preds[0].flatten()[0]
        visual_prob = multi_preds[1].flatten()[0]

        machining_status = "Yes" if machining_prob > 0.03 else "No"
        visual_status = "Passed" if visual_prob > 0.3 else "Failed"

        if tool_label == 'Worn':
            visual_status = "Failed"
            machining_status = "No"

        st.subheader("📝 Prediction Results")
        st.success(f"🔨 Tool Condition: {tool_label} ({tool_conf:.2f}% confidence)")
        st.info(f"🏱 Machining Finalized: {machining_status}")
        st.warning(f"🔍 Visual Inspection: {visual_status}")

        report_dict = {
            "Tool Condition": tool_label,
            "Confidence": f"{tool_conf:.2f}%",
            "Machining Finalized": f"{machining_status}",
            "Visual Inspection": f"{visual_status}"
        }

        csv_report = pd.DataFrame([report_dict])
        st.download_button("⬇️ Download CSV Report", csv_report.to_csv(index=False), file_name="tool_wear_report.csv", mime="text/csv")

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "", 16)

        pdf.set_fill_color(13, 78, 216)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, " CNC Tool Wear Prediction Report ", ln=True, align='C', fill=True)

        pdf.ln(10)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("DejaVu", "", 14)
        pdf.cell(0, 10, f"Tool Condition: {tool_label} ({tool_conf:.2f}% confidence)", ln=True)

        if machining_status == 'Yes':
            pdf.set_text_color(0, 102, 0)
        else:
            pdf.set_text_color(204, 0, 0)
        pdf.cell(0, 10, f"Machining Finalized: {machining_status} ({machining_prob:.2f})", ln=True)

        if visual_status == 'Passed':
            pdf.set_text_color(0, 102, 204)
        else:
            pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Visual Inspection: {visual_status} ({visual_prob:.2f})", ln=True)

        pdf_bytes = pdf.output(dest='S')
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin1')
        elif isinstance(pdf_bytes, bytearray):
              pdf_bytes = bytes(pdf_bytes)

        st.download_button("⬇️ Download Styled PDF Report", data=pdf_bytes, file_name="tool_wear_report.pdf", mime="application/pdf")
        st.success("✅ Predictions & Downloads Ready!")