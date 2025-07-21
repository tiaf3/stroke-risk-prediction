# Stroke Risk Prediction using Machine Learning

This project builds a robust machine learning pipeline to **predict stroke risk** based on clinical and lifestyle features.  
Achieved up to **97% accuracy** using **Random Forest** classifier.

---

## Tools & Libraries

- **Language**: Python  
- **Libraries**: Pandas, Scikit-learn, Matplotlib, Seaborn  
- **Models Used**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest  
- **Preprocessing**:
  - Missing value imputation
  - Label encoding
  - SMOTE for class balancing  
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC Curve

---

## Pipeline Overview

1. Data cleaning and imputation
2. Categorical encoding
3. Train-test split (70/30)
4. Class balancing with **SMOTE**
5. Model training & evaluation
6. Result visualization and comparison

---

## Results

| Model               | Accuracy | Precision | Recall     | F1 Score |
|---------------------|----------|-----------|------------|----------|
| Random Forest       | 96.64%   | **98.58%**| 94.68%     | **96.59%** |
| K-Nearest Neighbors | 92.84%   | 88.54%    | **98.50%** | 93.26%   |
| Logistic Regression | 77.99%   | 76.63%    | 80.91%     | 78.71%   |

- **Top Features**: Age, Hypertension, Smoking Status

> While **K-Nearest Neighbors** achieved the highest recall (98.5%), its lower precision reduced its overall F1-score.  
> **Random Forest** offered the most balanced performance across all metrics, with excellent precision (98.58%) and high recall, making it the most reliable model overall—especially for medical applications where both false positives and false negatives matter.  
> **Logistic Regression**, while significantly simpler, served as a good baseline but lacked the capacity to capture complex patterns in the data.

---

## Key Visualizations

### 🔹 Feature Importance  
<img src="images/feature_importance.png" width="600"/>

### 🔹 Accuracy Comparison  
<img src="images/model_accuracy.png" width="600"/>

### 🔹 ROC Curve  
<img src="images/roc_curve.png" width="600"/>

---

## Dataset

Data was obtained from the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/tiaf3/stroke_risk_prediction.git
   ```
2. Place the dataset CSV file inside the `data/` folder  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python stroke_prediction.py
   ```

---

## Status

✅ **Completed** – Academic/portfolio project  
Demonstrates strong ML pipeline design, performance tuning, and interpretability.
