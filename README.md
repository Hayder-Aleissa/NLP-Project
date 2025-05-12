# 📚 Arabic App Review Classification – NLP Term Project

This project focuses on the **automatic classification of Arabic app reviews** using various Natural Language Processing (NLP) techniques. It is divided into several parts, each covering a different experimental setup — from deep learning models built from scratch to advanced in-context learning using causal large language models (LLMs).

---

## 🔗 Dataset

We work with three labeled datasets provided by the course instructor:

- `AppReview-Multilabel.csv` – original annotated file (e.g., High/Medium/Low/No)
- `App Reviews-Multilabel.csv` – binarized multilabel format (e.g., 1/0)
- `App Reviews-SingleLabel-Multiclass.csv` – multiclass single-label format

**Data Splits:**
- 80% Training / 20% Testing
- 15% of the training data used for validation
- Random seed: `777`

---

## Project Breakdown

### Part A: Dataset Labelling
- Annotated raw Arabic app reviews with multilabels and multiclass tags.
- Organized dataset into structured CSVs for training and evaluation.

### Part B: Multiclass Classification (From Scratch)
- Built models using LSTM and Bidirectional LSTM.
- Compared:
  - Randomly initialized word embeddings
  - Pretrained FastText Arabic embeddings (`cc.ar.300.vec`)
- Addressed class imbalance using oversampling.
- Evaluated with metrics: Accuracy, F1 Score, Precision, Recall.

### Part C: Multilabel Classification (From Scratch)
- Adapted RNN-based models to handle multilabel outputs.
- Used the best embeddings from Part B.
- Performed in-depth classification analysis with macro/micro F1 scores.

### Part D: Multiclass Classification (Fine-tuned MLM)
- Fine-tuned pretrained Arabic language models (e.g., AraBERT or MARBERT) using HuggingFace Transformers.
- Used proper tokenization and padding tailored for Arabic.
- Analyzed classification performance and improvement strategies.

### Part E: Multilabel Classification (Fine-tuned MLM)
- Extended transformer-based approach to predict multiple labels per review.
- Thresholded sigmoid outputs and evaluated with multilabel metrics.
- Suggested improvements and explored label co-occurrence effects.

### Part F: Multiclass In-Context Learning (Causal LLMs)
- Performed **zero-shot** and **few-shot** classification using LLMs like GPT-3.5 (via OpenRouter).
- Designed and tested multilingual prompts (English and Arabic).
- Compared prompt strategies:
  - Basic prompts
  - Class-informed prompts
  - Chain-of-thought (CoT)
  - Example-based demonstrations
- Explored effects of:
  - Demonstration count (1/2/4/8/16)
  - Selection criteria (random, by class, similarity)
  - Prompt ordering (random vs. by label)

### Part G: Multilabel In-Context Learning (Causal LLMs)
- Similar to Part F, but adapted for multilabel tasks.
- Required structured outputs (e.g., JSON format).
- Designed prompts to handle multi-label prediction.
- Compared prompting strategies and scoring formats.
- Studied effects of task priming and label-aware prompting.

---

## ⚙️ Tools & Libraries

- **TensorFlow / Keras** – RNN-based model implementation
- **HuggingFace Transformers** – Pretrained model fine-tuning
- **OpenRouter API / GPT-3.5** – In-context learning via LLMs
- **scikit-learn** – Evaluation metrics and classification reports
- **pandas / matplotlib / seaborn** – Data analysis and visualization

---

## 📊 Evaluation & Analysis

- Each part includes:
  - Dataset statistics and distributions
  - Validation-based hyperparameter tuning
  - Final evaluation on the held-out test set
  - Macro & Micro F1, Precision, Recall, and Accuracy
- Some tasks include visualization of performance across configurations

---
## Course Information

- **Course**: ICS 472 – Natural Language Processing  
- **Institution**: King Fahd University of Petroleum and Minerals  
- **Term**: Second Semester, 2024–2025 (242)

## Licensing
This repository is developed as part of the KFUPM ICS 472 course (semester 242) and is intended solely for educational purposes.
