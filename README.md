# 📊 Customer Churn Prediction App

A complete, end-to-end Machine Learning pipeline and interactive web application built to predict customer churn using the telecom industry standards. This project implements advanced optimization strategies to handle real-world dataset challenges like class imbalance and model overfitting.

🌐 **Live Demo:** [https://customer-churn-prediction-ojzsyn85kiyxqfj6jpyjya.streamlit.app/]

---

## ✨ Features
* **Interactive Dashboard:** Input unique customer demographics, account statuses, and service subscriptions to get real-time churn predictions.
* **Risk Score Probabilities:** Displays how confident the model is regarding a customer's potential to leave or stay.
* **Proactive Recommendations:** Dynamically suggests business retention strategies based on calculated risk criteria.

---

## 🛠️ Machine Learning Optimizations Implemented

To meet professional standard requirements, the core notebook utilizes the following workflow:
1. **Downsampling:** Handled heavy class imbalance by downsampling the majority class using `sklearn.utils.resample` to avoid biased predictions.
2. **Stratified K-Fold Cross-Validation:** Utilized a 5-fold Stratified split ensuring that training iteration folds perfectly mimic real-world class distributions.
3. **Model Selection:** Compared performance metrics across multiple architectures, testing **Random Forest**, **Gradient Boosting**, and **XGBoost**.
4. **Hyperparameter Tuning (`GridSearchCV`):** Tuned parameters (`max_depth`, `min_samples_leaf`, `min_samples_split`) to achieve optimum bias-variance balance.
5. **Overfitting Countermeasures:** Restrained maximum decision-tree growth depths, shrinking the final train/test evaluation gap down to an incredibly tight **~5%**.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/gaurav25bm/Customer-Churn-Prediction.git](https://github.com/gaurav25bm/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
