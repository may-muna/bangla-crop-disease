# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import tempfile
import subprocess
import speech_recognition as sr
import gdown
import zipfile
import os
import queue
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# ----------------------------
# Google Drive FILE IDs
# ----------------------------
MODEL_PTH_ID = "1FodXFDpHPpIJWp2KKTks93E7hx5ID91U"
LABEL_ENCODER_ID = "1wU-u07LKw_oJVucfYzkJ8koNLUpNLkKD"
REMEDY_ID = "1NElCnlCJyZPRNEZ9LnvZGEFjX5_ZvLJO"
TOKENIZER_ZIP_ID = "1ngSnmJijllH-Y5SmmP6--7D-eWERqwhf"

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TOKENIZER_DIR = MODEL_DIR / "bert_lstm_final_tokenizer"
MODEL_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")
MAX_LEN = 128

# ----------------------------
# Download from Google Drive
# ----------------------------
def download_if_not_exists(file_id, out_path):
    if not out_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(out_path), quiet=False)

@st.cache_resource
def prepare_files():
    download_if_not_exists(MODEL_PTH_ID, MODEL_DIR / "bert_lstm_final_model.pth")
    download_if_not_exists(LABEL_ENCODER_ID, MODEL_DIR / "bert_lstm_final_label_encoder.pkl")
    download_if_not_exists(REMEDY_ID, MODEL_DIR / "bert_lstm_final_remedy.pkl")

    zip_path = MODEL_DIR / "tokenizer.zip"
    if not TOKENIZER_DIR.exists():
        gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_ZIP_ID}", str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

prepare_files()

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    label_encoder = joblib.load(MODEL_DIR / "bert_lstm_final_label_encoder.pkl")
    remedy_dict = joblib.load(MODEL_DIR / "bert_lstm_final_remedy.pkl")

    class BERT_LSTM_Model(nn.Module):
        def __init__(self, hidden_dim=128, dropout=0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
            self.lstm = nn.LSTM(
                self.bert.config.hidden_size,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, len(label_encoder.classes_))

        def forward(self, input_ids, attention_mask):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            lstm_out, _ = self.lstm(bert_out.last_hidden_state)
            x = self.dropout(lstm_out[:, 0, :])
            return self.fc(x)

    model = BERT_LSTM_Model().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "bert_lstm_final_model.pth", map_location=device))
    model.eval()

    return tokenizer, model, label_encoder, remedy_dict

tokenizer, model, label_encoder, remedy_dict = load_tokenizer_and_model()

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
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        pred = torch.argmax(model(**inputs), dim=1).item()

    disease = label_encoder.inverse_transform([pred])[0]
    remedy = remedy_dict.get(disease, "⚠️ কোনো প্রতিকার পাওয়া যায়নি।")
    return disease, remedy

# ----------------------------
# Audio upload transcription
# ----------------------------
def transcribe_audio(path):
    recog = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = recog.record(source)
    return recog.recognize_google(audio, language="bn-BD")

# ----------------------------
# Browser microphone (WebRTC)
# ----------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = queue.Queue()

    def recv_audio(self, frame: av.AudioFrame):
        self.frames.put(frame)
        return frame

# ----------------------------
# UI
# ----------------------------
st.title("🌾 ফসলের রোগ নির্ণয় সিস্টেম (বাংলা)")

method = st.radio(
    "ইনপুট পদ্ধতি নির্বাচন করুন:",
    ["✍ টেক্সট", "🎤 অডিও আপলোড", "🎙 মাইক্রোফোন"]
)

# ---- TEXT ----
if method == "✍ টেক্সট":
    text = st.text_area("রোগের লক্ষণ লিখুন:")
    if st.button("রোগ নির্ণয় করুন"):
        disease, remedy = predict_text(text)
        st.markdown(f"### 🦠 রোগ: **{disease}**")
        st.markdown(f"### 💊 প্রতিকার:\n{remedy}")

# ---- AUDIO UPLOAD ----
elif method == "🎤 অডিও আপলোড":
    audio_file = st.file_uploader("অডিও আপলোড করুন", type=["wav"])
    if st.button("রোগ নির্ণয় করুন") and audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.read())
            text = transcribe_audio(f.name)

        st.markdown(f"### 📝 শনাক্ত টেক্সট:\n{text}")
        disease, remedy = predict_text(text)
        st.markdown(f"### 🦠 রোগ: **{disease}**")
        st.markdown(f"### 💊 প্রতিকার:\n{remedy}")

# ---- MICROPHONE ----
elif method == "🎙 মাইক্রোফোন":
    st.info("🎙 ব্রাউজার মাইক্রোফোন ব্যবহার করুন")

    ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if st.button("রোগ নির্ণয় করুন"):
        if ctx.audio_processor and not ctx.audio_processor.frames.empty():
            frames = []
            while not ctx.audio_processor.frames.empty():
                frames.append(ctx.audio_processor.frames.get())

            pcm = b"".join([f.to_ndarray().tobytes() for f in frames])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(pcm)
                text = transcribe_audio(f.name)

            st.markdown(f"### 📝 শনাক্ত টেক্সট:\n{text}")
            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")
