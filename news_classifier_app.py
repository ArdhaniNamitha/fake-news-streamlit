import streamlit as st
import joblib
import json
import os
from datetime import datetime

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load or initialize history
HISTORY_FILE = "history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

# Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #232526, #414345);
            color: #fff;
        }
        .stApp {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(8px);
            padding: 2rem;
            border-radius: 20px;
        }
        .css-18ni7ap, .css-1d391kg {
            color: #eee !important;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)

# Input box
st.subheader("📌 Enter a news article:")
user_input = st.text_area("Write news content here 👇")

if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a news article!")
    else:
        # Preprocess input (lowercase)
        cleaned_input = user_input.lower()

        # Predict
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]

        # Debug output
        st.write(f"🧪 Raw prediction: {prediction}")
        st.write(f"📊 Probabilities → FAKE: {proba[0]:.4f}, REAL: {proba[1]:.4f}")
        st.write(f"🧠 Model classes: {model.classes_}")

        result_text = "✅ This news seems **REAL**." if prediction == 1 else "❌ This news seems **FAKE**."
        st.success(result_text)

        # Save to history
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": user_input,
            "result": "REAL" if prediction == 1 else "FAKE"
        })
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

# History viewer
st.markdown("---")
with st.expander("📜 View Classification History"):
    if not history:
        st.info("No history yet.")
    else:
        for item in reversed(history[-10:]):  # Show last 10
            st.markdown(f"""
                🕒 **{item['timestamp']}**  
                📝 *{item['text'][:80]}...*  
                🔎 **Result**: {item['result']}
                ---
            """)
