# Step 1: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# Set seeds
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)


# Set color palettes
colors1 = ["#3b3e79", "#e79898",  "#7575cf"]   # Main color palette (blue-purple + accents)
colors2 = ["#3b3e79", "#e79898"]               # Secondary palette (blue + pink)
colors3 = ["#3b3e79", "#5b9c7d", "#e75351"]    # For KDE and countplots (blue, green, red)


# Step 2: Load and Explore Dataset

dataset = pd.read_csv('data/healthcare-dataset-stroke.csv')
dataset.drop(['id'], axis=1, inplace=True)

# Fill missing BMI values with mean
dataset["bmi"].fillna(dataset["bmi"].mean(), inplace=True)

# Label encode categorical features
categorical_columns = dataset.select_dtypes(include='object').columns
encoder = LabelEncoder()
for col in categorical_columns:
    dataset[col] = encoder.fit_transform(dataset[col])


# Step 3: Exploratory Data Analysis (EDA)

# KDE plot for numerical features 
sns.kdeplot(dataset['age'], label='Age', shade=True, color=colors3[0])
sns.kdeplot(dataset['avg_glucose_level'], label='Glucose', shade=True, color=colors3[1])
sns.kdeplot(dataset['bmi'], label='BMI', shade=True, color=colors3[2])
plt.title("KDE Plot for Numerical Features")
plt.legend()
plt.tight_layout()
plt.show()

# Count plots for categorical features
features_to_plot = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'smoking_status', 'stroke']
plt.figure(figsize=(20, 24))
for i, col in enumerate(features_to_plot):
    plt.subplot(4, 2, i + 1)
    sns.countplot(data=dataset, x=col, palette=colors3)
    plt.title(f'Count Plot - {col}')
plt.tight_layout()
plt.show()

# Stroke distribution by categories
sns.catplot(x="gender", hue="stroke", data=dataset, kind="count", palette=colors2)
sns.catplot(x="ever_married", hue="stroke", data=dataset, kind="count", palette=colors2)
sns.catplot(x="work_type", hue="stroke", data=dataset, kind="count", palette=colors2)
sns.catplot(x="Residence_type", hue="stroke", data=dataset, kind="count", palette=colors2)
sns.catplot(x="smoking_status", hue="stroke", data=dataset, kind="count", palette=colors2)
sns.catplot(x="hypertension", hue="stroke", data=dataset, kind="count", palette=colors2)

# Heatmap of correlations 
plt.figure(figsize=(11,7), dpi=100)
sns.heatmap(dataset.corr(), annot=True, cmap=sns.light_palette(colors1[0], as_cmap=True), fmt=".2f")
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Age distribution for stroke vs non-stroke 
plt.hist(dataset[dataset['stroke'] == 0]['age'], bins=30, alpha=0.6, label='No Stroke', color=colors3[0])
plt.hist(dataset[dataset['stroke'] == 1]['age'], bins=20, alpha=0.5, label='Stroke', color=colors3[2])
plt.title('Age Distribution - Stroke vs No Stroke')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


# Step 4: Prepare Data for Modeling

X = dataset.drop('stroke', axis=1).values
y = dataset['stroke'].values

# Apply SMOTE to handle class imbalance
smote = SMOTE()
X, y = smote.fit_resample(X, y.ravel())

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 5: Train Models and Evaluate

# Logistic Regression
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train, y_train)
logistic_prediction = logistic_model.predict(X_test)
logistic_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=12, random_state=0)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)
rf_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_prediction = knn_model.predict(X_test)
knn_pred_prob = knn_model.predict_proba(X_test)[:, 1]


# Step 6: Print Metrics

def print_metrics(name, y_test, y_pred):
    print(f"\n{name} Metrics")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1 Score: {f1_score(y_test, y_pred) * 100:.2f}%")

print_metrics("Logistic Regression", y_test, logistic_prediction)
print_metrics("Random Forest", y_test, rf_prediction)
print_metrics("K-Nearest Neighbors", y_test, knn_prediction)


# Step 7: Accuracy Comparison Bar Plot

models_acc = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'KNN'],
    'Accuracy': [
        accuracy_score(y_test, logistic_prediction),
        accuracy_score(y_test, rf_prediction),
        accuracy_score(y_test, knn_prediction)
    ]
})

plt.figure(figsize=(10, 6))
ax1 = sns.barplot(data=models_acc, x='Model', y='Accuracy', color=colors1[0]) 
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)

# Add percentage labels inside the bars
for p in ax1.patches:
    acc = p.get_height()
    ax1.annotate(f'{acc:.2%}', (p.get_x() + p.get_width() / 2, acc - 0.05),
                 ha='center', va='top', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("images/model_accuracy.png", dpi=300)
plt.show()


# Step 8: ROC Curve 

models = ['Logistic Regression', 'Random Forest', 'KNN']
pred_probs = [logistic_pred_prob, rf_pred_prob, knn_pred_prob]
roc_auc_scores = []
roc_curves = []

for prediction in pred_probs:
    fpr, tpr, _ = roc_curve(y_test, prediction)
    roc_auc_scores.append(auc(fpr, tpr))
    roc_curves.append((fpr, tpr))

plt.figure(figsize=(8, 6))
for i in range(len(models)):
    plt.plot(
        roc_curves[i][0], roc_curves[i][1],
        color=colors1[i],
        lw=2,
        label=f'{models[i]} (AUC = {roc_auc_scores[i]:.2f})'
    )

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
plt.title('ROC Curve: Stroke Risk Prediction', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=300)
plt.show()


# Step 9: Feature Importance (Random Forest)

importances = rf_model.feature_importances_
feature_names = dataset.drop("stroke", axis=1).columns

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x='Importance', y='Feature', data=feature_df, color=colors1[0])  
plt.title('Top Features Influencing Stroke Risk', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.xlim(0, feature_df['Importance'].max() + 0.05)  # Add space for the label

# Add importance labels inside bars
for p in ax2.patches:
    width = p.get_width()
    ax2.annotate(f'{width:.2f}', (width - 0.01, p.get_y() + p.get_height() / 2),
                 ha='right', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=300)
plt.show()