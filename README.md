#  Sentiment Analysis on Airline Reviews: SVM vs. Stacked BiLSTM

##  Project Overview
Customer feedback is unstructured and messy. This project builds a Natural Language Processing (NLP) pipeline to classify airline reviews and predict whether a customer would recommend the airline (Yes/No).

I implemented and compared two distinct approaches:
1.  **Classical ML:** TF-IDF Vectorization with Support Vector Machine (SVM).
2.  **Deep Learning:** Custom Word2Vec Embeddings with a Stacked Bi-directional LSTM.

**Key Result:** The classical SVM model with SMOTE achieved the highest accuracy of **87.8%**, while the Deep Learning model reached **85.5%** with better generalization on nuanced text.

---

##  Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **NLP:** NLTK (Preprocessing), Word2Vec (Embeddings)
* **Machine Learning:** Scikit-learn (SVM, SMOTE, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn

---

## Methodology

### 1. Text Preprocessing & Cleaning
Implemented a robust cleaning pipeline using **NLTK**:
* Lowercasing and punctuation removal.
* Tokenization and Stopword removal.
* **Lemmatization** to reduce words to their base roots.

### 2. Traditional ML Approach (The Baseline)
* **Feature Engineering:** Used **TF-IDF** to vectorize the top 5,000 terms.
* **Imbalance Handling:** Applied **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance in the dataset.
* **Model Selection:** Benchmarked Logistic Regression, Random Forest, and SVM.
* **Optimization:** Used `GridSearchCV` for hyperparameter tuning.
* **Outcome:** SVM (Linear Kernel) + SMOTE yielded **87.8% Accuracy**.

### 3. Deep Learning Approach (The Neural Network)
* **Embeddings:** Trained custom **Word2Vec** embeddings on the corpus to capture semantic relationships specific to airline vocabulary.
* **Architecture:**
    * **Input Layer:** Pre-trained Word2Vec weights.
    * **Hidden Layers:** Stacked BiLSTM (2 layers) to capture sequential context in both directions.
    * **Regularization:** Dropout layers to prevent overfitting.
    * **Output:** Sigmoid activation for binary classification.
* **Training:** Implemented Early Stopping and Model Checkpointing in **PyTorch**.
* **Outcome:** Achieved **85.5% Accuracy**.

---

## Evaluation & Error Analysis
I conducted a deep dive into misclassified samples to understand model limitations:
* **Sarcasm:** "Great, another 3-hour delay" was often misclassified as positive.
* **Mixed Sentiment:** Reviews containing both praise (for crew) and complaints (for food) confused the models.

---

## Future Scope
* Implement **Transformer-based models (BERT/RoBERTa)** to better handle context and sarcasm.
* Explore **Attention Mechanisms** to visualize which words carry the most weight in the prediction.

## 👤 Author
**Sara Iqbal**
* [LinkedIn Profile]((https://www.linkedin.com/in/saraiqbaldata0602/))
* [GitHub Profile](https://github.com/sara-iqbal)
