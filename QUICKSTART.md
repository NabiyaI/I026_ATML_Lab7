# ATML Lab 7 - Quick Start Guide
## Encoder-Decoder Architecture with Attention

### 🎯 What Has Been Done

This complete implementation provides:

✅ **Simple Encoder-Decoder Model**
- Baseline architecture without attention
- BiLSTM encoder with LSTM decoder
- ~7.4M parameters
- Trained and saved

✅ **Encoder-Decoder with Attention**
- Bahdanau attention mechanism implemented
- Dynamic context-based decoding
- ~17.2M parameters
- Trained and saved

✅ **Comprehensive Comparison**
- Both models trained on identical datasets
- Same hyperparameters for fair comparison
- Performance metrics calculated
- Visualizations generated

✅ **Complete Documentation**
- Full lab submission template (LAB_SUBMISSION.md)
- Comprehensive README with explanations
- Code comments and docstrings
- Training logs and reports

---

### 📊 Key Results

**Training Performance:**
| Model | Train Loss | Val Loss | Improvement |
|-------|-----------|----------|------------|
| Simple | 1.3745 | 6.7644 | 64.64% |
| Attention | 0.7273 | 8.9854 | 81.24% |

**Architecture Comparison:**
| Aspect | Simple | Attention |
|--------|--------|-----------|
| Parameters | 7.4M | 17.2M |
| Context Type | Fixed | Dynamic |
| Attention | None | Bahdanau |
| Training Loss | Higher | Lower |
| Complexity | Lower | Higher |

---

### 📂 Files Organization

```
Project Root/
├── 📄 README.md - Project overview and documentation
├── 📄 LAB_SUBMISSION.md - Complete lab submission template
├── 📄 requirements.txt - Python dependencies
├── 📄 train.py - Main training script
├── 📄 utils.py - Utility functions and classes
├── 📁 models/
│   ├── simple_encoder_decoder.py - Baseline model
│   ├── encoder_decoder_attention.py - Attention model
│   └── __init__.py - Package initialization
├── 📁 results/ - Training outputs
│   ├── simple_encoder_decoder.pth - Saved model weights
│   ├── encoder_decoder_attention.pth - Saved model weights
│   ├── experiment_report.txt - Detailed report
│   ├── model_comparison.csv - Metrics table
│   ├── training_losses.png - Loss visualization
│   └── model_comparison_plot.png - Comparison plot
└── 📁 data/ - Data directory (empty)
```

---

### 🚀 How to Use

#### 1. **Review the Results**
```bash
# Read the comprehensive report
cat results/experiment_report.txt

# View the CSV comparison
less results/model_comparison.csv

# Open visualizations
open results/training_losses.png
open results/model_comparison_plot.png
```

#### 2. **Examine the Code**
```bash
# Look at model implementations
cat models/simple_encoder_decoder.py
cat models/encoder_decoder_attention.py

# Check utility functions
cat utils.py

# Review training script
cat train.py
```

#### 3. **Run the Complete Pipeline**
```bash
python3 train.py
```

#### 4. **Complete the Submission**
- Open `LAB_SUBMISSION.md`
- Fill in all sections (B.1, B.2, B.3, B.4, B.5)
- Add your analysis and conclusions
- Submit the completed document

---

### 📋 Lab Submission Tasks

Complete these sections in LAB_SUBMISSION.md:

**B.1: Implementation Overview**
- Project setup and data preparation
- Model specifications
- Hyperparameter justification

**B.2: Model Architecture Analysis**
- Detailed component descriptions
- Comparison of architectures
- Attention mechanism explanation
- Advantages and disadvantages

**B.3: Training Results**
- Performance metrics
- Loss curves analysis
- Convergence patterns
- Comparison table

**B.4: Interpretability & Attention Analysis**
- How attention works
- Bahdanau formula explanation
- Attention weights interpretation
- Use cases and benefits

**B.5: Conclusion**
- Summary of findings
- Significance of attention
- Verification of learning outcomes
- Suggested improvements

---

### 🔬 Experiment Summary

**Objective:** Compare encoder-decoder architecture with and without attention mechanism

**Dataset:** English-Hindi parallel sentences (15 pairs + 3 validation)

**Configuration:**
- Architecture: BiLSTM encoder + LSTM decoder
- Attention: Bahdanau multiplicative attention (attention model only)
- Training: 50 epochs, batch size 8, Adam optimizer

**Key Findings:**
1. Attention model achieves lower training losses (81.24% improvement)
2. Attention increases model capacity (2.3x more parameters)
3. Bahdanau attention enables dynamic context selection
4. Attention weights are interpretable for understanding model behavior
5. Trade-off between model complexity and generalization on small datasets

---

### 💾 Model Checkpoints

Both trained models are saved and can be loaded:

```python
import torch
from models import SimpleEncoderDecoder, EncoderDecoderWithAttention

# Load simple model
simple_model = SimpleEncoderDecoder(encoder, decoder)
simple_model.load_state_dict(torch.load('results/simple_encoder_decoder.pth'))

# Load attention model
attention_model = EncoderDecoderWithAttention(encoder, decoder)
attention_model.load_state_dict(torch.load('results/encoder_decoder_attention.pth'))
```

---

### 📚 References

1. **Attention Mechanism:**
   - Bahdanau et al., 2015 - "Neural Machine Translation by Jointly Learning to Align and Translate"

2. **Sequence-to-Sequence:**
   - Sutskever et al., 2014 - "Sequence to Sequence Learning with Neural Networks"

3. **Transformers:**
   - Vaswani et al., 2017 - "Attention is All You Need"

---

### ✅ Verification Checklist

Before submission, ensure:

- [ ] README.md read and understood
- [ ] LAB_SUBMISSION.md completed with all sections
- [ ] All Python code reviewed and understood
- [ ] Results visualized and interpreted
- [ ] B.1-B.5 sections well-written
- [ ] Conclusion reflects understanding of attention
- [ ] References cited appropriately

---

### 🎓 What You Should Understand

After reviewing this implementation:

✓ How encoder-decoder architectures work  
✓ Why fixed context vectors create bottlenecks  
✓ How Bahdanau attention solves alignment problems  
✓ Mathematical formulation of attention  
✓ Trade-offs between model complexity and performance  
✓ How to compare models fairly  
✓ Importance of attention in modern NLP  

---

### 🆘 Troubleshooting

**Q: Models are saved but I can't customize them?**  
A: Check that you have PyTorch installed and access to the model architecture definitions.

**Q: Can I run inference on the trained models?**  
A: Yes! Load the model and set it to eval mode, then pass new sequences through the model.

**Q: How do I visualize attention weights?**  
A: Modify the training script to return attention_weights and plot them as a heatmap.

**Q: What if I want to use my own dataset?**  
A: Modify the `load_english_hindi_data()` function in utils.py to load your data format.

---

### 📞 Next Steps

1. **Review** - Read through all documentation
2. **Understand** - Study the model implementations
3. **Complete** - Fill in LAB_SUBMISSION.md
4. **Verify** - Check all sections are complete
5. **Submit** - Submit the final lab document

---

**Status:** ✅ Project Complete and Ready for Submission  
**Generated:** March 2026  
**Lab:** Advanced Topics in Machine Learning (ATML)  
**Experiment:** 8 - Encoder-Decoder with Attention
