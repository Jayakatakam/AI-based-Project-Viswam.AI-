# 🌐 English to Telugu Neural Machine Translator

This project implements a **Neural Machine Translation (NMT)** system using **Facebook's mBART-50** model to translate English text into Telugu. It includes model fine-tuning, evaluation, and an interactive **Streamlit UI** for real-time translation.

---

## 🚀 Project Highlights

- 🔤 **Multilingual mBART-50**: Fine-tuned for English ➝ Telugu translation.
- 📚 **Custom Dataset**: Trained on `.jsonl` parallel corpora.
- 🧠 **Transformer-based Fine-tuning**: Using HuggingFace's `transformers` and `datasets`.
- 🖥️ **Streamlit UI**: Easy-to-use interface for end-users to enter English prompts and get Telugu translations.
- 💾 **Checkpoints & Tokenizer**: Saved for future inference and deployment.

---

## 📁 Project Structure

```bash
Eng-Tel/
├── data/
│   └── eng_telugu.jsonl         # Parallel dataset
├── mbart-en-te-checkpoints/     # Saved model after training
├── app.py                       # Streamlit application
├── train_translate.ipynb        # Jupyter notebook for training
├── tokenizer_config.json        # Tokenizer config
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 📦 Requirements

> Create a virtual environment before installing.

```bash
pip install -r requirements.txt
```

### Or manually:

```bash
pip install transformers datasets sentencepiece torch accelerate streamlit
```

---

## 🧠 Model Training

We fine-tune the `facebook/mbart-large-50-many-to-many-mmt` model on English-Telugu data.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="te_IN")
```

> Full training code available in `train_translate.ipynb`.

---

## 🎛️ Streamlit Interface

To run the app:

```bash
streamlit run app.py
```

### UI Features:

- English input box ✏️
- Telugu output with translation 🈳
- Clean, responsive layout

---

## 🧪 Example

**Input:**
```
How are you doing today?
```

**Output:**
```
మీరు ఈరోజు ఎలా ఉన్నారు?
```

---

## 🧰 Tools & Frameworks

| Tool             | Use                                |
|------------------|-------------------------------------|
| `transformers`   | mBART-50 model loading & training   |
| `datasets`       | Data preprocessing                  |
| `torch`          | PyTorch backend for model training  |
| `streamlit`      | UI frontend                         |
| `sentencepiece`  | Tokenizer support                   |

---

## 📈 Future Enhancements

- Add BLEU score evaluation on validation set
- Enable GPU-based Streamlit deployment
- Integrate translation history or speech-to-text

---

## 📜 License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

---

## 👨‍💻 Author

**Pavitr Swain**  
  
📫 [LinkedIn](https://www.linkedin.com/in/pavitr-swain/) | 🐙 [GitHub](https://github.com/Pavitr-Swain)