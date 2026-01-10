# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import tempfile
import subprocess
import sounddevice as sd
import numpy as np
import wavio
import speech_recognition as sr
import gdown
import zipfile

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TOKENIZER_DIR = MODEL_DIR / "bert_lstm_final_tokenizer"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# GOOGLE DRIVE FILE IDS
# ----------------------------
MODEL_PTH_ID = "1FodXFDpHPpIJWp2KKTks93E7hx5ID91U"
LABEL_ENCODER_ID = "1wU-u07LKw_oJVucfYzkJ8koNLUpNLkKD"
REMEDY_ID = "1NElCnlCJyZPRNEZ9LnvZGEFjX5_ZvLJO"
TOKENIZER_ZIP_ID = "1ngSnmJijllH-Y5SmmP6--7D-eWERqwhf"

# ----------------------------
# Download helper
# ----------------------------
def download_from_drive(file_id, output_path):
    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)

# ----------------------------
# Download model files
# ----------------------------
download_from_drive(
    MODEL_PTH_ID,
    MODEL_DIR / "bert_lstm_final_model.pth"
)

download_from_drive(
    LABEL_ENCODER_ID,
    MODEL_DIR / "bert_lstm_final_label_encoder.pkl"
)

download_from_drive(
    REMEDY_ID,
    MODEL_DIR / "bert_lstm_final_remedy.pkl"
)

tokenizer_zip = MODEL_DIR / "bert_lstm_final_tokenizer.zip"
if not TOKENIZER_DIR.exists():
    download_from_drive(TOKENIZER_ZIP_ID, tokenizer_zip)
    with zipfile.ZipFile(tokenizer_zip, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# ----------------------------
# Load tokenizer + model
# ----------------------------
device = torch.device("cpu")
MAX_LEN = 128

@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True
    )

    label_encoder = joblib.load(
        MODEL_DIR / "bert_lstm_final_label_encoder.pkl"
    )

    remedy_dict = joblib.load(
        MODEL_DIR / "bert_lstm_final_remedy.pkl"
    )

    class BERT_LSTM_Model(nn.Module):
        def __init__(self, hidden_dim=128, dropout=0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(
                "bert-base-multilingual-cased"
            )
            self.lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(
                hidden_dim * 2,
                len(label_encoder.classes_)
            )

        def forward(self, input_ids, attention_mask):
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            lstm_out, _ = self.lstm(bert_out.last_hidden_state)
            cls_token = self.dropout(lstm_out[:, 0, :])
            return self.classifier(cls_token)

    model = BERT_LSTM_Model().to(device)
    state_dict = torch.load(
        MODEL_DIR / "bert_lstm_final_model.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model, label_encoder, remedy_dict


tokenizer, model, label_encoder, remedy_dict = load_tokenizer_and_model()

# ----------------------------
# Prediction function
# ----------------------------
def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs, dim=1).item()

    disease = label_encoder.inverse_transform([pred])[0]
    remedy = remedy_dict.get(
        disease,
        "⚠️ কোনো প্রতিকার পাওয়া যায়নি।"
    )
    return disease, remedy

# ----------------------------
# Transcribe uploaded audio
# ----------------------------
def transcribe_bangla(audio_file):
    suffix = ".wav" if audio_file.type == "audio/wav" else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        input_audio_path = tmp.name

    wav_path = input_audio_path.replace(suffix, ".wav")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_audio_path,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            wav_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    recog = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recog.record(source)
        text = recog.recognize_google(audio, language="bn-BD")
    except:
        text = "⚠️ অডিও থেকে কিছু বুঝতে পারিনি।"

    return text

# ----------------------------
# Record audio (LOCAL ONLY)
# ----------------------------
def record_audio(duration=5, fs=16000):
    st.info("🔊 এখন কথা বলুন...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(tmp.name, audio, fs, sampwidth=2)
    return tmp.name

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌾 ফসলের রোগ নির্ণয় সিস্টেম (বাংলা)")

method = st.radio(
    "ইনপুট পদ্ধতি নির্বাচন করুন:",
    ["✍ টেক্সট", "🎤 অডিও আপলোড", "🎙 মাইক্রোফোন"]
)

# Text
if method == "✍ টেক্সট":
    text = st.text_area("রোগের লক্ষণ লিখুন:")
    if st.button("রোগ নির্ণয় করুন"):
        if text.strip():
            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")
        else:
            st.warning("⚠️ টেক্সট লিখুন।")

# Audio upload
elif method == "🎤 অডিও আপলোড":
    audio_file = st.file_uploader(
        "অডিও আপলোড করুন",
        type=["wav", "mp3"]
    )
    if st.button("রোগ নির্ণয় করুন") and audio_file:
        text = transcribe_bangla(audio_file)
        st.markdown(f"### 📝 শনাক্ত টেক্সট:\n{text}")
        if "⚠️" not in text:
            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")

# Microphone
elif method == "🎙 মাইক্রোফোন":
    st.warning("⚠️ Streamlit Cloud এ মাইক্রোফোন কাজ নাও করতে পারে")
    if st.button("রোগ নির্ণয় করুন"):
        try:
            wav = record_audio(10)
            recog = sr.Recognizer()
            with sr.AudioFile(wav) as source:
                audio = recog.record(source)
            text = recog.recognize_google(audio, language="bn-BD")
            st.markdown(f"### 📝 শনাক্ত টেক্সট:\n{text}")
            disease, remedy = predict_text(text)
            st.markdown(f"### 🦠 রোগ: **{disease}**")
            st.markdown(f"### 💊 প্রতিকার:\n{remedy}")
        except Exception as e:
            st.error(f"⚠️ সমস্যা হয়েছে: {e}")
