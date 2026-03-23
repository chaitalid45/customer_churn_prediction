# 📊 Customer Churn Prediction

This project predicts customer churn using machine learning techniques on an imbalanced dataset.

---

## 🚀 Problem Statement
Customer churn is a major business problem. The goal is to identify customers who are likely to leave the service.

---

## 📂 Project Structure
customer_churn/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── feature_extraction.ipynb
│ └── model_development.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ └── model_development.py
│
├── requirements.txt
└── README.md

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

---

## 🧠 Key Steps

### 1. Data Preprocessing
- Missing value handling
- Outlier treatment (log transformation)
- Duplicate removal

### 2. Feature Engineering
- Encoding categorical variables
- Dropping correlated features
- Creating new features (Total minutes, Total calls)

### 3. Handling Imbalanced Data
- Applied **SMOTE** to balance classes

### 4. Model
- Random Forest Classifier
- Threshold tuning to improve recall

---

## 📈 Results

| Metric | Score |
|------|------|
| Precision | ~0.79 |
| Recall | ~0.85 |
| F1 Score | ~0.82 |

👉 Focused on **high recall** to capture more churn customers.

---

## 🔥 Key Learnings

- Handling imbalanced datasets using SMOTE
- Avoiding data leakage
- Threshold tuning for business problems
- Feature engineering impact on performance

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
cd src
python model_development.py
├── requirements.txt
└── README.md
