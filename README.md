# I026_ATML_Lab7: Encoder-Decoder Architecture with Attention
## English-to-Hindi Machine Translation

## 📋 Project Overview

This lab implements an **Encoder-Decoder Architecture with Attention Mechanism** for English-to-Hindi Machine Translation. The experiment compares the performance of an attention-enhanced model with a baseline encoder-decoder model, demonstrating the significance of attention mechanisms in sequence-to-sequence tasks.

### 🎯 Learning Objectives

After completing this experiment, you will be able to:
1. Design and implement encoder-decoder architectures using LSTM
2. Implement Bahdanau attention mechanism from scratch
3. Train and evaluate sequence-to-sequence models
4. Compare model performance under identical conditions
5. Understand why attention mechanisms are crucial for neural machine translation
6. Interpret attention weights and model behavior

---

## 📁 Project Structure

```
I020_ATML_Lab7/
├── README.md                          # Project documentation
├── LAB_SUBMISSION.md                  # Complete lab submission template
├── requirements.txt                   # Python dependencies
├── train.py                          # Main training script
├── utils.py                          # Utility functions
├── models/
│   ├── simple_encoder_decoder.py    # Baseline model implementation
│   └── encoder_decoder_attention.py # Attention-based model implementation
├── results/
│   ├── simple_encoder_decoder.pth   # Saved model weights
│   ├── encoder_decoder_attention.pth # Saved model weights
│   ├── model_comparison.csv         # Comparison metrics
│   ├── experiment_report.txt        # Detailed text report
│   ├── training_losses.png          # Loss visualization
│   └── model_comparison_plot.png    # Comparison plot
└── data/                             # Data storage directory
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd I020_ATML_Lab7

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiment

```bash
# Run the complete training pipeline
python3 train.py
```

This will:
1. Load and prepare the English-Hindi dataset
2. Build vocabularies for both languages
3. Train the simple encoder-decoder model (50 epochs)
4. Train the encoder-decoder with attention (50 epochs)
5. Generate comparison metrics and visualizations
6. Print the comprehensive experiment report

**Expected Runtime:** ~5-10 minutes on CPU

---

## 📊 Experiment Results

### 🌐 Live Output

> **[▶ View Interactive Results Dashboard](https://nabiyai.github.io/I026_ATML_Lab7/view_results.html)**  
> An interactive HTML dashboard with training loss curves, model comparison charts, and a full performance table.

### Models Implemented

#### 1. Simple Encoder-Decoder (Baseline)
- **Architecture:** BiLSTM Encoder → Fixed Context Vector → LSTM Decoder
- **Parameters:** 7,405,617
- **Key Characteristics:**
  - All source information compressed into one context vector
  - Fixed context passed to decoder
  - No attention mechanism

**Performance:**
- Final Training Loss: 1.3745
- Final Validation Loss: 6.7644
- Training Loss Improvement: 64.64%

#### 2. Encoder-Decoder with Attention
- **Architecture:** BiLSTM Encoder → Bahdanau Attention → LSTM Decoder
- **Parameters:** 17,164,849
- **Key Characteristics:**
  - Dynamic context through attention mechanism
  - Multiplicative (Bahdanau) attention
  - Bidirectional encoder outputs used for attention

**Performance:**
- Final Training Loss: 0.7273
- Final Validation Loss: 8.9854
- Training Loss Improvement: 81.24%

### Comparison Results

| Metric | Simple Model | Attention Model | Difference |
|--------|-------------|-----------------|-----------|
| Training Loss Improvement | 64.64% | 81.24% | +16.60% |
| Final Training Loss | 1.3745 | 0.7273 | -47.09% |
| Average Training Loss | 2.3263 | 1.2622 | -45.77% |
| Parameters | 7.4M | 17.2M | +2.32x |

### Key Findings

1. ****Training Convergence:**
   - Attention model achieves significantly lower training losses
   - Superior optimization of training data
   - 81.24% improvement indicates better learning

2. **Parameters vs. Performance:**
   - Attention model has 2.3x more parameters
   - Larger model capacity enables better fitting
   - Trade-off between model size and learnability

3. **Generalization Characteristics:**
   - Smaller datasets: simple model may generalize better
   - Larger datasets: attention model typically excels
   - Overfitting behavior evident in validation losses

---

## 🔍 Technical Details

### Architecture Components

#### Encoder (BiLSTM)
```
Input Embeddings → BiLSTM (2 layers, 512 hidden dims)
                        ↓
              Encoder Outputs (full sequence)
                        ↓
              Hidden & Cell States (transformed to unidirectional)
```

#### Attention Mechanism (Bahdanau)
```
Decoder Hidden State
         ↓
    Attention Weights Calculation
    score(h_t, s_i) = v^T * tanh(W * [h_t; s_i])
         ↓
    Softmax Normalization
         ↓
    Context Vector = Weighted Sum of Encoder Outputs
```

#### Decoder (LSTM with Attention)
```
Previous Hidden State + Context Vector + Target Embedding
         ↓
       LSTM
         ↓
    LSTM Output + Context Vector
         ↓
      Dense Layer → Vocabulary Output
```

### Hyperparameters

- **Batch Size:** 8
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Epochs:** 50
- **Embedding Dimension:** 256
- **Hidden Dimension:** 512
- **Number of LSTM Layers:** 2
- **Dropout:** 0.5
- **Teacher Forcing Ratio:** 0.5
- **Max Sequence Length:** 20
- **Loss Function:** Cross-Entropy

---

## 📈 Visualizations

### Training Loss Curves
The `results/training_losses.png` shows:
- Simple model training and validation losses
- Attention model training and validation losses
- Convergence patterns for both architectures

### Model Comparison Plot
The `results/model_comparison_plot.png` shows:
- Side-by-side validation loss comparison
- Epoch-by-epoch performance tracking
- Relative performance differences

---

## 💡 Key Insights

### Why Attention Matters

1. **Information Bottleneck Solution**
   - Without Attention: All source info → one vector → decoder
   - With Attention: Decoder can access any encoder output at any time

2. **Alignment and Interpretability**
   - Attention weights show source-target word alignment
   - Enables understanding of model's translation decisions
   - Applicable to human linguistic analysis

3. **Handling Variable Lengths**
   - Better performance on long sequences
   - Attention weights prevent gradient vanishing
   - Encoder outputs are directly accessible

4. **Flexibility**
   - Different attention patterns for different tokens
   - Learned alignment rather than fixed mapping
   - Adaptive to data characteristics

---

## 🔧 Usage Examples

### Running with Custom Data

```python
from utils import load_english_hindi_data, Vocabulary, create_sequences

# Load your own data
english_texts, hindi_texts = load_english_hindi_data('path/to/data.csv')

# Build vocabularies
en_vocab = Vocabulary()
hi_vocab = Vocabulary()
en_vocab.build_vocabulary(english_texts)
hi_vocab.build_vocabulary(hindi_texts)

# Create sequences
en_seqs = create_sequences(english_texts, en_vocab)
hi_seqs = create_sequences(hindi_texts, hi_vocab)
```

### Inference Example

```python
from models.encoder_decoder_attention import EncoderDecoderWithAttention
from utils import load_model

# Load trained model
model = EncoderDecoderWithAttention(encoder, decoder)
load_model(model, 'results/encoder_decoder_attention.pth')

# Inference
model.eval()
with torch.no_grad():
    src = en_sequences[0].unsqueeze(0)  # Add batch dimension
    outputs, attention_weights = model(src, trg_sequences[0].unsqueeze(0))
    predictions = outputs.argmax(dim=-1)
```

---

## 📝 Lab Submission (Part B)

Complete the following tasks in your submission:

### B.1: Implementation Overview
- [ ] Describe the project architecture
- [ ] Explain data preparation steps
- [ ] List all hyperparameters used

### B.2: Model Architecture Analysis
- [ ] Compare simple encoder-decoder and attention models
- [ ] Explain each component's functionality
- [ ] Create architecture diagrams or descriptions

### B.3: Training Results
- [ ] Report training metrics for both models
- [ ] Create performance comparison table
- [ ] Analyze convergence patterns

### B.4: Interpretability Analysis
- [ ] Explain how attention mechanism works
- [ ] Describe Bahdanau attention formula
- [ ] Discuss attention weight interpretation

### B.5: Conclusion
- [ ] Summarize key findings
- [ ] Discuss significance of attention
- [ ] Suggest future improvements

---

## 📚 References

1. **Attention Mechanism:**
   - Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv preprint arXiv:1409.0473.

2. **Sequence-to-Sequence Models:**
   - Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks." arXiv preprint arXiv:1409.3215.

3. **Transformer Architecture:**
   - Vaswani, A., et al. (2017). "Attention is All You Need." In Advances in Neural Information Processing Systems (pp. 5998-6008).

4. **Practical Attention:**
   - Luong, M. T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation."

---

## 🎓 Learning Outcomes

After completing this lab, you will understand:

✓ How encoder-decoder architectures process sequential data  
✓ Why fixed context vectors create information bottlenecks  
✓ How attention mechanisms solve alignment problems  
✓ Mathematical foundations of Bahdanau attention  
✓ Trade-offs between model complexity and performance  
✓ How to train and evaluate sequence-to-sequence models  
✓ Practical importance of attention in modern NLP  

---

## ❓ FAQ

**Q: Why does the attention model have higher validation loss?**  
A: On small datasets, the more complex attention model may overfit. With larger datasets, it typically outperforms.

**Q: Can I use this for real translation?**  
A: This is a demonstration model. Real systems use much larger datasets, more complex architectures (Transformers), and additional techniques like beam search.

**Q: How do I interpret attention weights?**  
A: High attention weights indicate the source word is important for the current target word. Visualizing these reveals source-target alignments.

**Q: What's the difference between attention mechanisms?**  
A: Bahdanau (additive), Luong (multiplicative), and others differ in scoring functions. They're generally similar in practice.

---

## 📞 Support

For questions or issues:
1. Check the LAB_SUBMISSION.md for detailed explanations
2. Review the experiment report in `results/experiment_report.txt`
3. Examine the code comments in model implementations
4. Consult the referenced papers for theoretical background

---

**Last Updated:** March 2026  
**Status:** ✅ Complete and Ready for Submission
