# Stroke Risk Prediction using Machine Learning

This project builds a machine learning pipeline that predicts stroke risk using clinical and lifestyle features, Achieved  **97% accuracy** with the **Random Forest** model.

---

## Tools & Libraries
- **Python**: Pandas, Scikit-learn
- **Data preprocessing**: imputation, encoding, SMOTE
- **Models**: Logistic Regression, K-Nearest Neighbors, Random Forest
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC Curve
- **Visualization**: Matplotlib, Seaborn

---

## Project Steps
1. Data cleaning and missing value imputation
2. Label encoding of categorical variables
3. Split data into training and test sets (70/30)
4. Handled class imbalance using SMOTE
5. Trained and evaluated models:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (KNN)
6. Visualized results using bar plots and ROC curves

---

## Results
- Best model: Random Forest  
- Accuracy: `96.64%`
- F1-score: `96.59%`
- Key predictive features: **Age**, **Hypertension**, **Smoking Status**

Random Forest consistently outperformed other models in accuracy and F1-score, making it the most reliable choice overall.  
Although KNN achieved the highest recall 98.50%, Random Forest provided a better balance between precision and recall, along with clearer feature importance-critical for medical decision-making.

---

## Key Visualizations

### 🔹 Feature Importance 
![Feature Importance](images/feature_importance.png)

### 🔹 Model Performance Comparison
![Accuracy Bar Chart](images/model_accuracy.png)

### 🔹 ROC Curve for Model Comparison
![ROC Curve](images/roc_curve.png)

---

## Dataset
The data was taken from the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## How to Run
1. Clone the repository: `git clone https://github.com/tiaf3/stroke_risk_prediction.git`
2. Download the dataset from Kaggle and place the CSV file inside the data/ folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the script: `python stroke_prediction.py`

---

## Project Status
Completed - Academic project and portfolio showcase
