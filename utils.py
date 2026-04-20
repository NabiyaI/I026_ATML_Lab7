"""
Utility functions for data loading, preprocessing, and evaluation
"""
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import os

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Vocabulary:
    """Build vocabulary from text data"""
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
    
    def build_vocabulary(self, sentence_list):
        """Build vocabulary from list of sentences"""
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            words = sentence.lower().split()
            for word in words:
                frequencies[word] += 1
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text):
        """Convert text to indices"""
        tokenized_text = text.lower().split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]
    
    def __len__(self):
        return len(self.itos)


def load_english_hindi_data(data_path=None):
    """
    Load English-Hindi translation dataset
    If data_path is None, create a small sample dataset for demonstration
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        english_data = df['english'].tolist() if 'english' in df.columns else df.iloc[:, 0].tolist()
        hindi_data = df['hindi'].tolist() if 'hindi' in df.columns else df.iloc[:, 1].tolist()
    else:
        # Sample dataset for demonstration
        english_sentences = [
            "hello world",
            "how are you",
            "i love machine learning",
            "good morning",
            "thank you very much",
            "what is your name",
            "where do you live",
            "computer science is fun",
            "i am learning python",
            "machine translation is important",
            "natural language processing",
            "deep learning models",
            "artificial intelligence",
            "neural networks work well",
            "transformers are powerful",
        ]
        
        hindi_sentences = [
            "नमस्ते दुनिया",
            "आप कैसे हैं",
            "मुझे मशीन लर्निंग पसंद है",
            "सुप्रभात",
            "बहुत धन्यवाद",
            "आपका नाम क्या है",
            "आप कहाँ रहते हैं",
            "कंप्यूटर विज्ञान मजेदार है",
            "मैं पायथन सीख रहा हूँ",
            "मशीन अनुवाद महत्वपूर्ण है",
            "प्राकृतिक भाषा प्रसंस्करण",
            "गहरे सीखने के मॉडल",
            "कृत्रिम बुद्धिमत्ता",
            "तंत्रिका नेटवर्क अच्छी तरह काम करते हैं",
            "ट्रांसफर्मर शक्तिशाली हैं",
        ]
        
        english_data = english_sentences
        hindi_data = hindi_sentences
    
    return english_data, hindi_data


def preprocess_text(text, remove_special_chars=True):
    """Preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters if needed
    if remove_special_chars:
        text = re.sub(r"[^a-z\s']", '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_sequences(text_data, vocab, max_length=20):
    """Convert text data to sequences"""
    sequences = []
    for text in text_data:
        preprocessed = preprocess_text(text, remove_special_chars=False)
        seq = [2] + vocab.numericalize(preprocessed) + [3]  # Add BOS and EOS
        
        # Pad or truncate
        if len(seq) < max_length + 2:
            seq += [0] * (max_length + 2 - len(seq))
        else:
            seq = seq[:max_length + 2]
        
        sequences.append(seq)
    
    return torch.LongTensor(sequences)


def calculate_bleu_score(reference_tokens, hypothesis_tokens):
    """
    Calculate BLEU score (simplified version)
    """
    from collections import Counter
    
    # Count n-grams (using unigrams for simplicity)
    ref_counter = Counter(reference_tokens)
    hyp_counter = Counter(hypothesis_tokens)
    
    # Calculate precision
    overlap = sum((ref_counter & hyp_counter).values())
    precision = overlap / max(len(hypothesis_tokens), 1)
    
    # Brevity penalty
    if len(hypothesis_tokens) < len(reference_tokens):
        brevity_penalty = np.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1))
    else:
        brevity_penalty = 1.0
    
    bleu = brevity_penalty * precision
    return bleu


def evaluate_translations(predictions, references):
    """Evaluate translation quality"""
    metrics = {
        'bleu_scores': [],
        'lengths_match': []
    }
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        bleu = calculate_bleu_score(ref_tokens, pred_tokens)
        metrics['bleu_scores'].append(bleu)
        metrics['lengths_match'].append(len(pred_tokens) == len(ref_tokens))
    
    return {
        'avg_bleu': np.mean(metrics['bleu_scores']),
        'avg_bleu_std': np.std(metrics['bleu_scores']),
        'length_match_ratio': np.mean(metrics['lengths_match'])
    }


def save_model(model, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Load model checkpoint"""
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model
