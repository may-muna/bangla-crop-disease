# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import joblib
import tempfile
import soundfile as sf
import speech_recognition as sr
import gdown
import zipfile
import io

# Install streamlit_audio_recorder if not installed:
# pip install streamlit-audio-recorder

from streamlit_audio_recorder import audio_recorder

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TOKENIZER_DIR = MODEL_DIR / "bert_lstm_final_tokenizer"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# Google Drive files
# ----------------------------
files = {
    "bert_lstm_final_tokenizer.zip": "1Ub6VHt4f3V4FyLIrYn6I_e1ayu3yerTn",
    "bert_lstm_final_remedy.pkl":      "1xwMe9VTdePuw_qRkEZoooWevxk0XcNtc",
    "bert_lstm_final_model.pth":       "1zWtlLUMA9UM1ggatNbzPgsTMB1FR5MNf",
    "bert_lstm_final_label_encoder.pkl":"1suK3wLB6iV57pM8lQ5PyJFpN6D8ddP1d",
    "bert_lstm_checkpoint.pth":        "1R1KxCUfznN0taWVsTrwRyu8tVrXbqYmH"
}

# ----------------------------
# Download files if not exists
# ----------------------------
for fname, file_id in files.items():
    dest_path = MODEL_DIR / fname
    if not dest_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Downloading {fname} from Google Drive...")
        gdown.download(url, str(dest_path), quiet=False)
        if fname.endswith(".zip"):
            st.info("Extracting tokenizer zip...")
            with zipfile.ZipFile(dest_path, 'r') as z:
                z.extractall(TOKENIZER_DIR)
            st.success("Tokenizer extraction done.")

# ----------------------------
# Load tokenizer, label encoder, remedy
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
label_encoder = joblib.load(MODEL_DIR / "bert_lstm_final_label_encoder.pkl")
remedy_dict = joblib.load(MODEL_DIR / "bert_lstm_final_remedy.pkl")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cpu")

# ----------------------------
# BERT+LSTM Model
# ----------------------------
class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=len(label_encoder.classes_), dropout=0.3):
        super().__init__()
        BERT_DIR = MODEL_DIR / "bert-base-multilingual-cased"
        self.bert = BertModel.from_pretrained(str(BERT_DIR).replace("\\", "/"), local_files_only=True)
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        lstm_out = self.dropout(lstm_out)
        cls_output = lstm_out[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# ----------------------------
# Load model
# ----------------------------
model = BERT_LSTM_Model(num_classes=len(label_encoder.classes_)).to(device)
state = torch.load(MODEL_DIR / "bert_lstm_final_model.pth", map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# ----------------------------
# Prediction
# ----------------------------
MAX_LEN = 128

def predict_text_raw(text):
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
# Bangla speech-to-text
# ----------------------------
def transcribe_bangla_audio_bytes(audio_bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.flush()

    r = sr.Recognizer()
    with sr.AudioFile(tmp.name) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="bn-BD")
        except Exception:
            text = "⚠️ অডিও বুঝতে পারিনি। আবার চেষ্টা করুন।"
    return text

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌾 ফসলের রোগ নির্ণয় সিস্টেম (বাংলা)")

input_method = st.radio("ইনপুট নির্বাচন করুন:", ["লিখিত লক্ষণ", "কণ্ঠ ইনপুট (মাইক্রোফোন)"])

if input_method == "লিখিত লক্ষণ":
    symptoms = st.text_area("রোগের লক্ষণ লিখুন (বাংলায়):")
    if st.button("পাঠান"):
        if not symptoms.strip():
            st.warning("⚠️ দয়া করে রোগের লক্ষণ লিখুন।")
        else:
            disease, remedy = predict_text_raw(symptoms)
            st.markdown(f"🦠 **সনাক্ত রোগ:** {disease}")
            st.markdown(f"💊 **প্রতিকার:** {remedy}")

elif input_method == "কণ্ঠ ইনপুট (মাইক্রোফোন)":
    audio_bytes = audio_recorder()
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        text = transcribe_bangla_audio_bytes(audio_bytes)
        st.markdown(f"📝 **সনাক্ত টেক্সট:** {text}")
        disease, remedy = predict_text_raw(text)
        st.markdown(f"🦠 **সনাক্ত রোগ:** {disease}")
        st.markdown(f"💊 **প্রতিকার:** {remedy}")
