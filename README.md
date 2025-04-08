# ✈️ GenAI Airline Review Analysis & Customer Booking Prediction

This project combines Natural Language Processing (NLP) and Machine Learning (ML) to analyze airline reviews and predict customer booking behavior. It's structured into two core tasks:

---

## 🧠 Project 1: British Airways Review Analysis

### 🔍 Goal
Scrape real-world customer reviews from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways), perform sentiment analysis, and extract latent topics using LDA.

### ✅ Features
- Web scraping using `requests` and `BeautifulSoup`
- Sentiment analysis using `TextBlob`
- Word cloud generation for visualizing frequent terms
- Latent Dirichlet Allocation (LDA) for topic modeling
- Exportable, cleaned dataset (`cleaned_airline_reviews.csv`)

### 📊 Output Examples
- **Sentiment Pie Chart**
- **Word Cloud**
- **Top Words Per Topic (LDA)**

### 📂 Output Files
- `BA_reviews.csv`
- `cleaned_airline_reviews.csv`

---

## 🔮 Project 2: Customer Booking Prediction

### 🎯 Goal
Predict whether a customer will complete a flight booking based on trip features using machine learning.

### ✅ Features
- Class imbalance handled via SMOTE
- One-hot encoding of categorical features
- Feature scaling using `StandardScaler`
- Model training using `RandomForestClassifier`
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation using:
  - Classification Report
  - Confusion Matrix
  - ROC AUC Score
  - Precision-Recall AUC
  - Feature Importance Plot

### 🛠️ Pipeline
```python
Pipeline([
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])
