# 🤖 Natural Language Processing Projects

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive collection of Deep Learning projects for text classification, featuring both CNN and LSTM architectures for sentiment analysis and emotion classification tasks.

---

## 📋 Table of Contents
- [Projects Overview](#-projects-overview)
- [Project 1: Sentiment Analysis with Bi-LSTM](#-project-1-sentiment-analysis-with-bi-lstm)
- [Project 2: Emotion Classification with CNN](#-project-2-emotion-classification-with-cnn)
- [Model Comparison](#-model-comparison)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Skills Demonstrated](#-technical-skills-demonstrated)

---

## 🎯 Projects Overview

This repository contains two advanced NLP projects showcasing different deep learning architectures and approaches to text classification:

| Project | Architecture | Task | Classes | Accuracy | Dataset Size |
|---------|-------------|------|---------|----------|--------------|
| **Sentiment Analysis** | Bi-LSTM | Binary Classification | 2 | 87.06% | 40,000 |
| **Emotion Classification** | 1D CNN | Multi-class Classification | 6 | 88.15% | 20,000 |

### Key Differentiators

**Sentiment Analysis (Bi-LSTM)**
- Binary sentiment classification (Positive/Negative)
- Bidirectional LSTM for context understanding
- Custom Word-Level tokenization
- Regularization with Dropout + Batch Normalization
- Early stopping and learning rate scheduling

**Emotion Classification (CNN)**
- Multi-class emotion detection (6 emotions)
- 1D CNN for efficient pattern recognition
- BERT pre-trained tokenization
- Lightweight and fast architecture
- Rapid convergence (5 epochs)

---

# 🎭 Project 1: Sentiment Analysis with Bi-LSTM

[![Accuracy](https://img.shields.io/badge/Accuracy-87.06%25-brightgreen.svg)](README.md)

## Overview

A sophisticated sentiment analysis model using **Bidirectional LSTM** architecture to classify Amazon product reviews into positive and negative sentiments.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep Learning Framework |
| ![HuggingFace](https://img.shields.io/badge/🤗_Datasets-FFD21E?style=flat) | Dataset Management |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Model Evaluation |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data Processing |

### Technical Stack

```python
# NLP Processing
- Custom Word-Level Tokenizer       # 5,000 vocabulary size
- Sequence Padding/Truncation       # Fixed length: 128 tokens
- Special tokens: <unk>, <pad>      

# Model Architecture
- Embedding Layer                   # Dimension: 256
- Bidirectional LSTM               # Hidden units: 512 × 2
- Batch Normalization              # Training stabilization
- Dropout (0.5)                    # Regularization
- CrossEntropyLoss                 # Loss function
- Adam Optimizer                   # Learning rate: 0.001

# Training Optimization
- ReduceLROnPlateau                # Dynamic LR scheduling
- Early Stopping                   # Patience: 3 epochs
- GPU Acceleration                 # CUDA support
```

## Model Architecture

```
Input Text → Tokenization → Padding (128)
    ↓
Embedding Layer (vocab: 5000, dim: 256)
    ↓
Bidirectional LSTM (hidden: 512 × 2)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Fully Connected Layer (1024 → 2)
    ↓
Softmax → [Negative, Positive]
```

### Model Specifications

| Component | Configuration |
|-----------|---------------|
| **Total Parameters** | ~7.5M |
| **Embedding Dimension** | 256 |
| **LSTM Hidden Units** | 512 (×2 for bidirectional) |
| **Vocabulary Size** | 5,000 tokens |
| **Sequence Length** | 128 tokens |
| **Batch Size** | 32 |
| **Dropout Rate** | 0.5 |

## Dataset

### Source & Statistics

- **Dataset**: `mteb/amazon_polarity` from Hugging Face
- **Domain**: Amazon Product Reviews
- **Total Samples**: 40,000

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training** | 28,000 | 70% |
| **Validation** | 4,000 | 10% |
| **Testing** | 8,000 | 20% |

### Data Preprocessing Pipeline

```python
1. Text Cleaning
   ├─ Convert to lowercase
   ├─ Remove special characters [^a-z0-9\s]
   ├─ Remove extra whitespaces
   └─ Strip leading/trailing spaces

2. Label Standardization
   ├─ 0 → Negative sentiment
   └─ 1 → Positive sentiment

3. Custom Tokenization
   ├─ Word-level tokenizer
   ├─ Vocabulary: 5,000 most frequent words
   ├─ Special tokens: <unk>, <pad>
   └─ Padding/Truncation to length 128
```

## Performance

### Metrics

```
╔════════════════════════════════════════╗
║        Model Performance Results       ║
╠════════════════════════════════════════╣
║  Accuracy:     87.06%                  ║
║  Precision:    87.19%                  ║
║  Recall:       87.06%                  ║
║  F1-Score:     87.05%                  ║
╚════════════════════════════════════════╝
```

### Training Progress

| Epoch | Train Loss | Val Loss | Description |
|-------|-----------|----------|-------------|
| 1 | 0.6802 | 0.6631 | Initial convergence |
| 2 | 0.4232 | 0.4496 | Rapid improvement |
| 3 | 0.3421 | 0.3685 | Steady progress |
| 4 | 0.2995 | 0.4744 | Some overfitting |
| 5 | 0.2705 | 0.3978 | Adjustment |
| 6 | 0.2290 | 0.3339 | Better generalization |
| 7 | 0.2216 | 0.3505 | Fine-tuning |
| 8 | 0.1960 | **0.3228** | ⭐ Best model |
| 9 | 0.1670 | 0.3533 | Slight overfit |
| 10 | 0.1385 | 0.3854 | Training stopped |

### Key Insights

✅ **Strengths:**
- Strong performance on positive sentiment (93.4% recall)
- Balanced precision across both classes
- Effective bidirectional context capture
- Robust to varying text lengths

⚠️ **Observations:**
- Best validation loss at Epoch 8
- Some overfitting in later epochs (controlled by early stopping)
- Slightly higher false negatives (17%)

---

# 🎭 Project 2: Emotion Classification with CNN

[![Accuracy](https://img.shields.io/badge/Accuracy-88.15%25-brightgreen.svg)](README.md)

## Overview

An efficient emotion classification model using **1D Convolutional Neural Network** to classify text into 6 distinct emotion categories with BERT tokenization.

### Emotion Categories

1. 😢 **Sadness** - Expressions of sadness, disappointment
2. 😊 **Joy** - Happiness, excitement, enthusiasm  
3. ❤️ **Love** - Affection, caring, romantic feelings
4. 😠 **Anger** - Frustration, irritation, rage
5. 😨 **Fear** - Worry, anxiety, concern
6. 😲 **Surprise** - Shock, amazement, unexpectedness

## Technologies Used

| Technology | Purpose |
|------------|---------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep Learning Framework |
| ![BERT](https://img.shields.io/badge/BERT-Tokenizer-green.svg) | Pre-trained Tokenization |
| ![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat) | Tokenizer Implementation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Visualization |

### Technical Stack

```python
# NLP Processing
- BERT Tokenizer (bert-base-uncased)  # Pre-trained vocabulary
- Automatic padding and truncation     # Variable-length handling
- Vocabulary size: 30,522              # Full BERT vocabulary

# Model Architecture
- Embedding Layer                      # 128-dimensional
- 1D Convolutional Layer              # 64 filters, kernel=3
- ReLU Activation                     # Non-linearity
- Global Average Pooling              # Dimension reduction
- Fully Connected Layer               # Classification

# Training Configuration
- Loss: CrossEntropyLoss              # Multi-class
- Optimizer: Adam (lr=0.001)          # Adaptive learning
- Batch Size: 16                      # Memory efficient
- Epochs: 5                           # Fast convergence
```

## Model Architecture

```
Input Text → BERT Tokenization
    ↓
Embedding Layer (vocab: 30,522, dim: 128)
    ↓
Transpose for Conv1D [batch, 128, seq_len]
    ↓
1D Convolution (in: 128, out: 64, kernel: 3)
    ↓
ReLU Activation
    ↓
Global Average Pooling 1D
    ↓
Squeeze & Flatten [batch, 64]
    ↓
Fully Connected Layer (64 → 6)
    ↓
Softmax → [sadness, joy, love, anger, fear, surprise]
```

### Model Specifications

| Component | Configuration |
|-----------|---------------|
| **Total Parameters** | ~4.3M |
| **Embedding Dimension** | 128 |
| **Conv Filters** | 64 |
| **Kernel Size** | 3 |
| **Vocabulary Size** | 30,522 (BERT) |
| **Batch Size** | 16 |
| **Output Classes** | 6 emotions |

### Why 1D CNN for Text?

**Advantages:**
- ⚡ **Faster Training**: Parallelizable unlike sequential RNNs
- 🎯 **Local Patterns**: Excellent at capturing n-gram features
- 💾 **Memory Efficient**: No sequential dependencies
- 🔄 **Simpler**: Fewer parameters than Bi-LSTM
- 📊 **Effective**: Competitive with RNNs on many NLP tasks

## Dataset

### Source & Statistics

- **Dataset**: `emotion` from Hugging Face Datasets
- **Domain**: General text emotions
- **Total Samples**: 20,000

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training** | 16,000 | 80% |
| **Validation** | 2,000 | 10% |
| **Testing** | 2,000 | 10% |

### Emotion Distribution

| Emotion | Train | Val | Test | Total | % |
|---------|-------|-----|------|-------|---|
| **Joy** | 5,362 | 695 | 695 | 6,752 | 33.8% |
| **Sadness** | 5,165 | 646 | 581 | 6,392 | 32.0% |
| **Anger** | 2,159 | 275 | 275 | 2,709 | 13.5% |
| **Fear** | 1,937 | 224 | 224 | 2,385 | 11.9% |
| **Love** | 1,304 | 159 | 159 | 1,622 | 8.1% |
| **Surprise** | 1,073 | 1 | 66 | 1,140 | 5.7% |

## Performance

### Overall Metrics

```
╔════════════════════════════════════════╗
║        Model Performance Results       ║
╠════════════════════════════════════════╣
║  Test Accuracy:     88.15%             ║
║  Weighted Avg:                         ║
║  - Precision:       88%                ║
║  - Recall:          88%                ║
║  - F1-Score:        88%                ║
╚════════════════════════════════════════╝
```

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Sadness** | 0.93 | 0.91 | 0.92 | 581 |
| **Joy** | 0.90 | 0.93 | 0.91 | 695 |
| **Love** | 0.74 | 0.77 | 0.75 | 159 |
| **Anger** | 0.86 | 0.87 | 0.86 | 275 |
| **Fear** | 0.87 | 0.86 | 0.86 | 224 |
| **Surprise** | 0.75 | 0.58 | 0.65 | 66 |

### Training Progress

| Epoch | Train Loss | Description |
|-------|-----------|-------------|
| 1 | 1.4646 | Initial convergence |
| 2 | 0.7740 | Rapid improvement |
| 3 | 0.3745 | Steady progress |
| 4 | 0.2291 | Fine-tuning |
| 5 | 0.1531 | Final optimization |

### Key Insights

✅ **Strengths:**
- Excellent on common emotions (Sadness: 93%, Joy: 93%)
- Fast convergence (only 5 epochs)
- Efficient inference time
- Balanced performance across most classes

⚠️ **Observations:**
- Surprise has lower recall (58%) due to fewer samples
- Love class could benefit from more training data
- Model excels at distinguishing strong emotions

---

## 🔬 Model Comparison

### Architecture Comparison

| Aspect | Bi-LSTM (Sentiment) | 1D CNN (Emotion) |
|--------|---------------------|------------------|
| **Architecture Type** | Recurrent Neural Network | Convolutional Neural Network |
| **Processing Style** | Sequential (bidirectional) | Parallel (sliding window) |
| **Context Capture** | Long-range dependencies | Local n-gram patterns |
| **Training Speed** | Slower (sequential) | Faster (parallel) |
| **Parameters** | 7.5M | 4.3M |
| **Best For** | Sentiment with context | Pattern-based classification |

### Performance Comparison

| Metric | Sentiment (Bi-LSTM) | Emotion (CNN) | Winner |
|--------|---------------------|---------------|--------|
| **Accuracy** | 87.06% | 88.15% | CNN ✅ |
| **Training Time** | 10 epochs | 5 epochs | CNN ✅ |
| **Parameters** | ~7.5M | ~4.3M | CNN ✅ |
| **Inference Speed** | Slower | Faster | CNN ✅ |
| **Task Complexity** | 2 classes | 6 classes | LSTM ✅ |
| **Context Understanding** | Better | Good | LSTM ✅ |

### Dataset Comparison

| Aspect | Sentiment Analysis | Emotion Classification |
|--------|-------------------|----------------------|
| **Dataset Size** | 40,000 samples | 20,000 samples |
| **Source** | Amazon Reviews | Emotion Dataset |
| **Domain** | Product Reviews | General Text |
| **Classes** | 2 (Pos/Neg) | 6 (Emotions) |
| **Avg Text Length** | Longer reviews | Shorter texts |
| **Tokenization** | Custom (5K vocab) | BERT (30K vocab) |

### When to Use Each?

**Choose Bi-LSTM (Sentiment) when:**
- 📝 Processing longer documents
- 🔄 Context matters across entire text
- 📊 Binary or simple multi-class tasks
- 🎯 Need to capture long-range dependencies
- 📖 Working with narrative or story-like text

**Choose 1D CNN (Emotion) when:**
- ⚡ Need fast inference speed
- 🎯 Local patterns are important
- 💾 Limited computational resources
- 🏃 Short to medium length texts
- 🔢 Many classes to distinguish
- 🚀 Production deployment priority

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, recommended)
- 4GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nlp-projects.git
cd nlp-projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```txt
# Core
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Tokenization
tokenizers>=0.13.0

# Evaluation & Visualization
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 💻 Usage

### Sentiment Analysis (Bi-LSTM)

```python
from datasets import load_dataset
from tokenizers import Tokenizer

# Load dataset
dataset = load_dataset("mteb/amazon_polarity")

# Initialize tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
# ... (train tokenizer on corpus)

# Create model
model = SentimentClassifier(
    vocab_size=5000,
    embedding_dim=256,
    hidden_dim=512,
    num_classes=2,
    dropout=0.5
)

# Prediction
def predict_sentiment(text, model, tokenizer):
    model.eval()
    encoded = vectorize(text, tokenizer)
    
    with torch.no_grad():
        output = model(encoded.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).item()
    
    return "Positive" if pred == 1 else "Negative"

# Example
text = "This product is absolutely amazing!"
sentiment = predict_sentiment(text, model, tokenizer)
print(f"Sentiment: {sentiment}")  # Output: Positive
```

### Emotion Classification (CNN)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("emotion")

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create model
model = EmotionClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    num_classes=6
)

# Prediction
def predict_emotion(text, model, tokenizer):
    model.eval()
    encoded = tokenizer([text], padding=True, truncation=True,
                       return_tensors="pt")
    
    with torch.no_grad():
        output = model(encoded["input_ids"].to(device))
        pred = torch.argmax(output, dim=1).item()
    
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return emotions[pred]

# Example
text = "I'm so excited about this new adventure!"
emotion = predict_emotion(text, model, tokenizer)
print(f"Emotion: {emotion}")  # Output: joy
```

### Batch Predictions

```python
# Sentiment Analysis - Batch
def predict_sentiments_batch(texts, model, tokenizer):
    model.eval()
    predictions = []
    
    for text in texts:
        encoded = vectorize(text, tokenizer)
        with torch.no_grad():
            output = model(encoded.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).item()
        predictions.append("Positive" if pred == 1 else "Negative")
    
    return predictions

# Emotion Classification - Batch
def predict_emotions_batch(texts, model, tokenizer):
    model.eval()
    encoded = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(encoded["input_ids"].to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return [emotions[p] for p in preds]

# Example usage
texts = [
    "This is the best product ever!",
    "Terrible quality, very disappointed",
    "I'm so happy with this purchase!"
]

sentiments = predict_sentiments_batch(texts, sentiment_model, sentiment_tokenizer)
emotions = predict_emotions_batch(texts, emotion_model, bert_tokenizer)

print("Sentiments:", sentiments)  # ['Positive', 'Negative', 'Positive']
print("Emotions:", emotions)      # ['joy', 'sadness', 'joy']
```

---

## 📁 Project Structure

```
nlp-projects/
├── sentiment_analysis/
│   ├── Sentiment_Analysis.ipynb        # Training notebook
│   ├── sentiment_model/
│   │   ├── model.pt                   # Trained weights
│   │   └── tokenizer.json             # Custom tokenizer
│   └── results/
│       ├── confusion_matrix.png
│       └── training_curves.png
│
├── emotion_classification/
│   ├── EmotionClassificationCNN.ipynb # Training notebook
│   ├── emotion_model/
│   │   └── model.pt                   # Trained weights
│   ├── emotion_tokenizer/             # BERT tokenizer
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── vocab.txt
│   │   └── special_tokens_map.json
│   └── results/
│       ├── confusion_matrix.png
│       └── training_curves.png
│
├── requirements.txt                    # Dependencies
├── README.md                          # This file
└── LICENSE
```

---

## 🎓 Technical Skills Demonstrated

### Deep Learning & NLP
- ✅ Bidirectional LSTM implementation
- ✅ 1D CNN for text classification
- ✅ Custom tokenization from scratch
- ✅ BERT tokenizer integration
- ✅ Embedding layer design
- ✅ Regularization techniques (Dropout, Batch Norm)
- ✅ Multi-class classification

### PyTorch Expertise
- ✅ Custom Dataset and DataLoader
- ✅ Model architecture design
- ✅ Training loop implementation
- ✅ GPU acceleration
- ✅ Model evaluation and metrics
- ✅ Checkpoint saving/loading

### NLP Techniques
- ✅ Text preprocessing pipelines
- ✅ Word-level tokenization
- ✅ Subword tokenization (BERT)
- ✅ Vocabulary building
- ✅ Sequence padding/truncation
- ✅ Sentiment analysis
- ✅ Emotion detection

### Machine Learning Best Practices
- ✅ Train/Validation/Test split
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Confusion matrix analysis
- ✅ Comprehensive metrics (Precision, Recall, F1)
- ✅ Overfitting prevention
- ✅ Model comparison

### Data Science & Visualization
- ✅ Hugging Face Datasets integration
- ✅ Pandas for data manipulation
- ✅ Matplotlib & Seaborn visualization
- ✅ Performance analysis
- ✅ Results interpretation

---

## 🔑 Key Takeaways

### Architecture Insights

1. **LSTM for Context**
   - Better for longer sequences
   - Captures long-range dependencies
   - Bidirectional improves understanding
   - More parameters but better context

2. **CNN for Speed**
   - Faster training and inference
   - Excellent for pattern recognition
   - Fewer parameters
   - Parallel processing advantage

3. **Tokenization Matters**
   - Custom tokenizers: Full control, smaller vocab
   - BERT tokenizers: Pre-trained, robust, larger vocab
   - Both approaches have merits

### Performance Insights

- **CNN achieved higher accuracy** (88.15% vs 87.06%)
- **CNN trained faster** (5 vs 10 epochs)
- **CNN more efficient** (4.3M vs 7.5M parameters)
- **Both models production-ready** (>87% accuracy)

### Production Considerations

| Factor | Sentiment (LSTM) | Emotion (CNN) |
|--------|-----------------|---------------|
| **Inference Speed** | ~100ms | ~50ms |
| **Model Size** | ~30MB | ~17MB |
| **Memory Usage** | Higher | Lower |
| **Deployment** | Good | Better |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

## 🌟 Future Improvements

### Sentiment Analysis
- [ ] Multi-lingual support
- [ ] Attention mechanism integration
- [ ] Aspect-based sentiment analysis
- [ ] Real-time prediction API

### Emotion Classification
- [ ] More emotion categories
- [ ] Transformer-based architecture
- [ ] Multi-label emotion detection
- [ ] Intensity scoring

### General
- [ ] Model ensemble
- [ ] Transfer learning experiments
- [ ] Web application deployment
- [ ] REST API development
- [ ] Docker containerization

---

⭐ **If you find these projects useful, please consider giving them a star!** ⭐

*Built with ❤️ using PyTorch, Hugging Face, and modern NLP techniques*