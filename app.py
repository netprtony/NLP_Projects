import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import re
import os

# ================= Định nghĩa mô hình SentimentClassifier =================
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=256, hidden_dim=512, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(x)
        x = torch.cat((hidden[0], hidden[1]), dim=1)
        x = self.batch_norm(x)
        x = self.fc(x)
        return x

# ================= Định nghĩa mô hình EmotionClassifier =================
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

# ================= Hàm tiền xử lý văn bản =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# ================= Hàm vector hóa cho SentimentClassifier =================
def vectorize_sentiment(sentence, tokenizer, sequence_length=128):
    output = tokenizer.encode(sentence)
    ids = output.ids[:sequence_length]
    padding = [0] * (sequence_length - len(ids))
    ids = ids + padding
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# ================= Hàm vector hóa cho EmotionClassifier =================
def vectorize_emotion(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# ================= Hàm dự đoán cho SentimentClassifier =================
def predict_sentiment(text, model, tokenizer, sequence_length=128):
    model.eval()
    processed_text = preprocess_text(text)
    input_ids = vectorize_sentiment(processed_text, tokenizer, sequence_length)
    with torch.no_grad():
        outputs = model(input_ids)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, dim=1)
    confidence = probs[0, pred.item()].item() * 100
    sentiment_label = "Positive 😊" if pred.item() == 1 else "Negative 😔"
    return sentiment_label, confidence

# ================= Hàm dự đoán cho EmotionClassifier =================
def predict_emotion(sentence, model, tokenizer, device):
    model.eval()
    encoded = vectorize_emotion([sentence], tokenizer)
    encoded = {key: val.to(device) for key, val in encoded.items()}
    with torch.no_grad():
        output = model(encoded["input_ids"])
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).cpu().item()
    confidence = probs[0, prediction].item() * 100
    emotions = ["Sadness 😢", "Joy 😄", "Love 😍", "Anger 😣", "Fear 😱", "Surprise 😮"]
    return emotions[prediction], confidence

# ================= Kiểm tra file và tải mô hình =================
# Đường dẫn đến file mô hình và tokenizer
sentiment_model_path = "./sentiment_model/model.pt"
sentiment_tokenizer_path = "./sentiment_model/tokenizer.json"
emotion_model_path = "./emotion_classifier_model/emotion_classifier.pth"
emotion_tokenizer_path = "./emotion_classifier_model"

# Kiểm tra sự tồn tại của file
st.write("Checking file paths...")
if not os.path.exists(sentiment_model_path):
    st.error(f"🚨 Sentiment model file not found at: {sentiment_model_path}")
    st.stop()
if not os.path.exists(sentiment_tokenizer_path):
    st.error(f"🚨 Sentiment tokenizer file not found at: {sentiment_tokenizer_path}")
    st.stop()
if not os.path.exists(emotion_model_path):
    st.error(f"🚨 Emotion model file not found at: {emotion_model_path}")
    st.stop()
if not os.path.exists(emotion_tokenizer_path):
    st.error(f"🚨 Emotion tokenizer directory not found at: {emotion_tokenizer_path}")
    st.stop()

# Kiểm tra các file tokenizer cần thiết (không yêu cầu config.json)
required_tokenizer_files = [
    os.path.join(emotion_tokenizer_path, "tokenizer.json"),
    os.path.join(emotion_tokenizer_path, "vocab.txt"),
    os.path.join(emotion_tokenizer_path, "special_tokens_map.json")
]
st.write(f"Tokenizer files in {emotion_tokenizer_path}: {os.listdir(emotion_tokenizer_path)}")
for file in required_tokenizer_files:
    if not os.path.exists(file):
        st.error(f"🚨 Missing tokenizer file: {file}")
        st.stop()

# Tải SentimentClassifier
vocab_size = 5000
embedding_dim = 256
hidden_dim = 512
num_classes = 2
sequence_length = 128

sentiment_model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
sentiment_model.load_state_dict(torch.load(sentiment_model_path, map_location=torch.device('cpu')))
sentiment_tokenizer = Tokenizer.from_file(sentiment_tokenizer_path)

# Tải EmotionClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    emotion_tokenizer = PreTrainedTokenizerFast.from_pretrained(emotion_tokenizer_path)
except Exception as e:
    st.error(f"🚨 Error loading emotion tokenizer: {str(e)}")
    st.error("Please ensure the tokenizer directory contains tokenizer.json, vocab.txt, and special_tokens_map.json.")
    st.stop()

emotion_model = EmotionClassifier(vocab_size=emotion_tokenizer.vocab_size, embedding_dim=128, num_classes=6)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=device))
emotion_model.to(device)

# ================= Danh sách câu ví dụ =================
example_sentences = [
    "I'm so excited about this new adventure!",
    "I feel so sad after losing my favorite book.",
    "I love spending time with my family.",
    "I'm furious about the unfair treatment at work!",
    "I'm terrified of speaking in front of a large crowd.",
    "What a surprise to see you here!",
    "This movie was absolutely amazing!",
    "I feel hopeless about the future.",
    "My heart is full of joy today!",
    "I'm so scared of the dark sometimes."
]

# ================= Giao diện Streamlit =================
st.title("🎭 Sentiment & Emotion Analysis")
st.write("Enter an English sentence or select an example below to predict its sentiment (Positive/Negative) and emotion (Sadness/Joy/Love/Anger/Fear/Surprise):")

# Dropdown để chọn câu ví dụ
selected_example = st.selectbox("Choose an example sentence:", [""] + example_sentences)
user_input = st.text_area("Or enter your own sentence:", value=selected_example, height=100, placeholder="e.g., I'm so excited about this new adventure!")

if st.button("Predict 🔍"):
    if user_input:
        # Dự đoán với SentimentClassifier
        sentiment_pred, sentiment_conf = predict_sentiment(user_input, sentiment_model, sentiment_tokenizer, sequence_length)
        st.success(f"**Sentiment Prediction**: {sentiment_pred} ({sentiment_conf:.2f}%)")

        # Dự đoán với EmotionClassifier
        emotion_pred, emotion_conf = predict_emotion(user_input, emotion_model, emotion_tokenizer, device)
        st.success(f"**Emotion Prediction**: {emotion_pred} ({emotion_conf:.2f}%)")
    else:
        st.error("⚠️ Please enter or select a sentence to predict!")

# Thêm thông tin phụ
st.markdown("---")
st.markdown("### About the Models")
st.markdown("- **Sentiment Model**: Predicts whether the input is Positive 😊 or Negative 😔 using a Bidirectional LSTM.")
st.markdown("- **Emotion Model**: Classifies the input into one of six emotions (Sadness 😢, Joy 😄, Love 😍, Anger 😣, Fear 😱, Surprise 😮) using a Conv1D architecture.")
st.markdown("---")
st.markdown("**Note**: Ensure all model and tokenizer files are correctly placed in the specified directories.")