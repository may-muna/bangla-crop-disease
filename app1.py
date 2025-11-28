# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import joblib
import tempfile
import speech_recognition as sr
import gdown
import zipfile
import shutil
import os

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TOKENIZER_DIR = MODEL_DIR / "bert_lstm_final_tokenizer"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# Google Drive downloadable links
# ----------------------------
files = {
    # Convert folder → ZIP download
    "bert_lstm_final_tokenizer.zip":
        "https://drive.google.com/uc?export=download&id=1XPgrRdWfO3B50EaWAcR9zIKHNlCbnA6-",

    "bert_lstm_final_remedy.pkl":      "1xwMe9VTdePuw_qRkEZoooWevxk0XcNtc",
    "bert_lstm_final_model.pth":       "1zWtlLUMA9UM1ggatNbzPgsTMB1FR5MNf",
    "bert_lstm_final_label_encoder.pkl":"1suK3wLB6iV57pM8lQ5PyJFpN6D8ddP1d",
}

# ----------------------------
# Download files + extract tokenizer
# ----------------------------
for fname, link in files.items():
    dest = MODEL_DIR / fname

    if not dest.exists():
        st.info(f"Downloading {fname} ...")

        # gdown supports full URL, so use it directly
        gdown.download(link, str(dest), quiet=False)

        if fname.endswith(".zip"):
            st.info("Extracting tokenizer...")

            if TOKENIZER_DIR.exists():
                shutil.rmtree(TOKENIZER_DIR)

            TOKENIZER_DIR.mkdir(exist_ok=True)

            with zipfile.ZipFile(dest, 'r') as z:
                z.extractall(TOKENIZER_DIR)

            st.success("Tokenizer extracted successfully!")

# Debug on Streamlit Cloud
st.write("Tokenizer files:", list(TOKENIZER_DIR.glob("*")))

# ----------------------------
# Load tokenizer & labels
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    local_files_only=True
)

label_encoder = joblib.load(MODEL_DIR / "bert_lstm_final_label_encoder.pkl")
remedy_dict = joblib.load(MODEL_DIR / "bert_lstm_final_remedy.pkl")

device = torch.device("cpu")

# ----------------------------
# BERT+LSTM Model
# ----------------------------
class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=len(label_encoder.classes_), dropout=0.3):
        super().__init__()
        BERT_DIR = MODEL_DIR / "bert-base-multilingual-cased"

        self.bert = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True)

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_out.last_hidden_state)
        cls_token = self.dropout(lstm_out[:, 0, :])
        return self.classifier(cls_token)

# ----------------------------
# Load trained model
# ----------------------------
model = BERT_LSTM_Model().to(device)
state = torch.load(MODEL_DIR / "bert_lstm_final_model.pth", map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

MAX_LEN = 128

# ----------------------------
# Prediction
# ----------------------------
def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        pred = torch.argmax(logits, dim=1).item()

    disease = label_encoder.inverse_transform([pred])[0]
    remedy = remedy_dict.get(disease, "⚠️ কোনো প্রতিকার পাওয়া যায়নি।")

    return disease, remedy

# ----------------------------
# Audio → Bangla text
# ----------------------------
def transcribe_bangla(audio_file):
    if audio_file is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        filename = tmp.name

    recog = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recog.record(source)

    try:
        text = recog.recognize_google(audio, language="bn-BD")
    except:
        text = "⚠️ অডিও থেকে কিছু বুঝতে পারিনি।"

    return text

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌾 ফসলের রোগ নির্ণয় সিস্টেম (বাংলা)")

method = st.radio("ইনপুট পদ্ধতি নির্বাচন করুন:", ["✍ টেক্সট", "🎤 অডিও আপলোড"])

if method == "✍ টেক্সট":
    text = st.text_area("রোগের লক্ষণ লিখুন:")

    if st.button("রোগ নির্ণয় করুন"):
        if not text.strip():
            st.warning("⚠️ টেক্সট লিখুন।")
        else:
            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")

else:
    audio = st.file_uploader("অডিও আপলোড করুন", type=["wav", "mp3"])

    if st.button("রোগ নির্ণয় করুন"):
        if audio is None:
            st.warning("⚠️ অডিও আপলোড করুন।")
        else:
            text = transcribe_bangla(audio)
            st.markdown(f"### 📝 শনাক্ত টেক্সট:\n{text}")

            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")
