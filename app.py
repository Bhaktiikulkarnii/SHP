# ============================================================
# SMART HEALTH PREDICTION SYSTEM - FULL FEATURED WITH FIXED VOICE INPUT
# ============================================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from fuzzywuzzy import process
import speech_recognition as sr
import pyttsx3
from fpdf import FPDF
import csv
import io

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Smart Health Predictor", page_icon="🩺", layout="wide")

# ---------------------- Load & Prepare Data ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diseases.csv")
    df["symptoms"] = df["symptoms"].apply(lambda x: [s.strip().lower() for s in x.split(",")])
    return df

df = load_data()

# Train model
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["disease"]
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# ---------------------- Session State ----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "matched_symptoms" not in st.session_state:
    st.session_state.matched_symptoms = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

# ---------------------- Sidebar User Info ----------------------
st.sidebar.header("👤 User Info")
age = st.sidebar.number_input("Age", 0, 120, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
smoking = st.sidebar.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
allergies = st.sidebar.text_input("Allergies (comma-separated)")

# ---------------------- Personalized Precautions ----------------------
def personalized_precautions(age, gender, smoking, allergies):
    tips = []
    if age >= 60:
        tips.append("Regular checkups and monitor BP and sugar levels.")
    if gender.lower() == "female":
        tips.append("Maintain bone health and iron levels.")
    if smoking.lower() == "smoker":
        tips.append("Avoid smoking to improve lung/heart health.")
    if allergies:
        tips.append(f"Avoid allergens: {allergies}")
    if not tips:
        tips.append("Maintain a healthy lifestyle: diet, exercise, hydration.")
    return " | ".join(tips)

# ---------------------- Offline Multi-Language Mapping ----------------------
lang_symptom_map = {
    # Fever
    "fever": "fever",
    "उबेर": "fever",
    "ताप": "fever",
    "बुखार": "fever",
    # Headache
    "headache": "headache",
    "डोकेदुखी": "headache",
    "सिरदर्द": "headache",
    # Loss of taste
    "loss of taste": "loss of taste",
    "चवीचा अभाव": "loss of taste",
    "चव न लागणे": "loss of taste",
    "स्वाद का नुकसान": "loss of taste",
    # Nausea
    "nausea": "nausea",
    "मळमळ": "nausea",
    "उलटी येणे": "nausea",
    "मतली": "nausea",
    # Rash
    "rash": "rash",
    "पुरळ": "rash",
    "त्वचेवर पुरळ": "rash",
    "दाने": "rash",
    # Runny nose
    "runny nose": "runny nose",
    "नाक सर्द": "runny nose",
    "नाक वाहणे": "runny nose",
    "नाक बहना": "runny nose",
    # Sensitivity to light
    "sensitivity to light": "sensitivity to light",
    "प्रकाशाची संवेदनशीलता": "sensitivity to light",
    "प्रकाश के प्रति संवेदनशीलता": "sensitivity to light",
    # Shortness of breath
    "shortness of breath": "shortness of breath",
    "श्वास घेताना त्रास": "shortness of breath",
    "श्वास घेणे कठीण": "shortness of breath",
    "साँस लेने में कठिनाई": "shortness of breath",
    # Stomach pain
    "stomach pain": "stomach pain",
    "पोटदुखी": "stomach pain",
    "पोटात दुखणे": "stomach pain",
    "पेट दर्द": "stomach pain",
    # Weakness
    "weakness": "weakness",
    "अशक्तपणा": "weakness",
    "थकवा": "weakness",
    "कमज़ोरी": "weakness"
}

# ---------------------- Symptom Input ----------------------
st.title("🩺 Smart Health Prediction System")
st.write("Select your symptoms or speak them to get predictions, precautions, and Aurangabad doctor details.")

symptom_list = sorted(list(mlb.classes_))

# Multi-language text input
lang_input = st.text_input("Enter symptoms (English/Marathi/Hindi):")
if lang_input:
    translated_input = []
    for word in lang_input.split(","):
        word = word.strip()
        translated_input.append(lang_symptom_map.get(word, word.lower()))
    # Fuzzy match
    for word in translated_input:
        match = process.extractOne(word, symptom_list)
        if match and match[1] > 70:
            if match[0] not in st.session_state.selected_symptoms:
                st.session_state.selected_symptoms.append(match[0])

# ---------------------- Voice Input ----------------------
st.subheader("🎤 Or Speak Symptoms")
if st.button("Record Symptom"):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            audio = r.listen(source)
        voice_text = r.recognize_google(audio)
        st.session_state.voice_text = voice_text
        st.success(f"You said: {voice_text}")

        # Offline mapping + fuzzy matching
        matched_symptoms = []
        for word in voice_text.split(","):
            word = word.strip()
            eng_sym = lang_symptom_map.get(word, word.lower())
            match = process.extractOne(eng_sym, symptom_list)
            if match and match[1] > 70:
                if match[0] not in st.session_state.selected_symptoms:
                    st.session_state.selected_symptoms.append(match[0])
                    matched_symptoms.append(match[0])

        if matched_symptoms:
            st.info(f"Matched Symptoms: {matched_symptoms}")
        else:
            st.warning("No symptoms matched from your speech.")

    except Exception as e:
        st.error(f"Could not recognize your voice. Error: {e}")

st.write("**Selected Symptoms:**", st.session_state.selected_symptoms)

# ---------------------- Text-to-Speech Function ----------------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------------- Predict Button ----------------------
if st.button("Predict"):
    if not st.session_state.selected_symptoms:
        st.error("Please enter or speak at least one symptom.")
    else:
        input_vec = mlb.transform([st.session_state.selected_symptoms])
        proba = clf.predict_proba(input_vec)[0]
        top_indices = proba.argsort()[::-1][:3]

        st.subheader("🧾 Prediction Results:")
        emergency_flag = False
        report_data = []

        for idx in top_indices:
            disease = clf.classes_[idx]
            confidence = proba[idx]*100
            info = df[df["disease"]==disease].iloc[0]

            # Emergency alert
            critical_symptoms = ["chest pain", "shortness of breath", "severe headache", "fever"]
            if any(sym in st.session_state.selected_symptoms for sym in critical_symptoms):
                emergency_flag = True

            # Display results
            st.markdown(f"### 🦠 {disease} ({confidence:.1f}% confidence)")
            st.markdown(f"**Precautions:** {info['precautions']}")
            st.markdown(f"**Personalized Tips:** {personalized_precautions(age, gender, smoking, allergies)}")
            st.markdown(f"**Specialist:** {info['specialist']}")
            st.markdown(f"**Doctor:** {info['doctor_name']}")
            st.markdown(f"**Contact:** {info['doctor_contact']}")
            st.markdown(f"**Address:** {info['doctor_address']}")
            st.write("---")

            # TTS
            tts_text = f"Predicted disease is {disease} with confidence {confidence:.1f} percent"
            speak_text(tts_text)

            # Store in report
            report_data.append({
                "Disease": disease,
                "Confidence": f"{confidence:.1f}%",
                "Precautions": info['precautions'],
                "Personalized Tips": personalized_precautions(age, gender, smoking, allergies),
                "Specialist": info['specialist'],
                "Doctor": info['doctor_name'],
                "Contact": info['doctor_contact'],
                "Address": info['doctor_address']
            })

        if emergency_flag:
            st.error("⚠️ Emergency Alert! Critical symptoms detected. Seek immediate medical attention!")

        # ---------------------- Session History ----------------------
        st.session_state.history.append({"symptoms": st.session_state.selected_symptoms, "predictions": report_data})

        st.subheader("📊 Symptom History (This Session)")
        for i, h in enumerate(st.session_state.history):
            st.markdown(f"**Entry {i+1}:** Symptoms: {h['symptoms']}")
            for r in h['predictions']:
                st.markdown(f"- {r['Disease']} ({r['Confidence']})")

        # ---------------------- Export Report ----------------------
        st.subheader("💾 Export Report")

        # CSV
        csv_file = io.StringIO()
        keys = report_data[0].keys()
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(report_data)
        st.download_button(
            label="Download CSV Report",
            data=csv_file.getvalue(),
            file_name="prediction_report.csv",
            mime="text/csv"
        )

        # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Smart Health Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(0, 10, txt=f"User Info: Age={age}, Gender={gender}, Smoking={smoking}, Allergies={allergies}", ln=True)
        pdf.ln(5)
        for r in report_data:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt=f"Disease: {r['Disease']} ({r['Confidence']})", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, txt=f"Precautions: {r['Precautions']}")
            pdf.multi_cell(0, 8, txt=f"Personalized Tips: {r['Personalized Tips']}")
            pdf.multi_cell(0, 8, txt=f"Specialist: {r['Specialist']}")
            pdf.multi_cell(0, 8, txt=f"Doctor: {r['Doctor']}")
            pdf.multi_cell(0, 8, txt=f"Contact: {r['Contact']}")
            pdf.multi_cell(0, 8, txt=f"Address: {r['Address']}")
            pdf.ln(5)

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )
