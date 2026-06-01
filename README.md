# Heart-Disease-ML-Prediction
Machine Learning pipeline evaluating Logistic Regression, KNN, and Random Forest architectures to predict heart disease risk from clinical metrics.

# Heart Disease Prediction: Machine Learning Evaluation Project

This repository hosts a formal machine learning pipeline executed for the PlutoAcademy AI & ML Internship Program. The project preprocesses real-world clinical metrics, engineers scaled features, trains 3 unique classification algorithms, and reviews performance markers to pinpoint the safest diagnostic model.

## Workspace & Data Context
* **Google Colab Notebook:** [Access Code & Workspace Here](https://colab.research.google.com/drive/1HpvyOBb0Q8mj08KJBUR_hwNtpkePknD-?usp=sharing)
* **Dataset Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 1. Preprocessing & Feature Engineering Choices
* **Data Cleansing:** Checked for null matrices and safely removed redundant data vectors to keep evaluation arrays clean.
* **Stratified 80/20 Splitting:** Enforced strict class balancing across train/test splits (`stratify=y`) so minority baseline outcomes are accurately accounted for.
* **Z-score Standardization:** Transformed independent variables using a standard scaler. This neutralizes units of measure (e.g., age vs. blood pressure metrics) across KNN distance evaluations and Logistic Regression weights.

---

## 2. Machine Learning Comparison Summary
The following matrix evaluates performance metrics recorded over the identical testing partition:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Best)** | **0.8033** | **0.8000** | **0.8485** | **0.8235** |
| **K-Nearest Neighbors** | 0.7869 | 0.7778 | 0.8485 | 0.8116 |
| **Random Forest** | 0.7541 | 0.7647 | 0.7879 | 0.7761 |

---

## 3. Best Model Choice & Final Written Conclusion
The **Logistic Regression** model emerged as the top-performing architecture for this dataset, achieving the highest overall Accuracy (80.33%) and F1-Score (82.35%). In clinical diagnostics, optimizing for a high **Recall** (84.85%) is absolutely critical to minimize False Negatives, ensuring that patients with heart disease are not mistakenly sent home undiagnosed. Logistic Regression performed best here because the underlying relationship between features like maximum heart rate, cholesterol, and chest pain type exhibits a strong linear log-odds structure, which allows it to generalize beautifully on smaller tabular samples without overfitting like Random Forest.

---

## 4. Analytical Surprise
> What surprised me the most was that **Logistic Regression outpaced the Random Forest Classifier** by roughly 5% across both Accuracy and F1-Score. In machine learning, there is a common assumption that complex ensemble models always outperform simpler baseline algorithms. However, this result perfectly demonstrates the principle of Occam's Razor in data science: when dealing with limited medical data vectors, highly complex models can easily overfit to noise, whereas a well-regularized linear model can often map the true clinical boundary much more effectively.
