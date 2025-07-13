# 🧠 Sentiment Analyser using Naive Bayes

This project performs **Sentiment Analysis** on product reviews using **Naive Bayes classification**. It classifies each review as **Positive**, **Negative**, or **Neutral**. The project includes a clean preprocessing pipeline, model training with TF-IDF vectorization, and optional UI using **Gradio**.

---

## 📂 Dataset

- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Size:** 568,454 reviews
- **Attributes Used:**  
  - `Text` – the review  
  - `Score` – numeric rating (1 to 5)

---

## 🧪 Preprocessing

- Lowercasing text  
- Removing punctuation  
- Removing stopwords  
- Token filtering (alphabetic words only)
- TF-IDF vectorization using `ngram_range=(1,2)` and `max_features=5000`

---

## 🧮 Sentiment Mapping

| Score | Sentiment |
|-------|-----------|
| 1–2   | Negative  |
| 3     | Neutral   |
| 4–5   | Positive  |

To ensure balance, the model is trained on 4,000 samples per class.

---

## 📈 Model Details

- **Algorithm:** Multinomial Naive Bayes (`sklearn.naive_bayes`)
- **Accuracy:** >90% (for binary classes) or ~75–85% (for balanced three-class setup)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

