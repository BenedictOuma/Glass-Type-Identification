# Glass Identification - Forensic Analysis Project

## Project Overview

This project simulates a forensic analyst's role in a criminal investigation unit. When glass fragments are found at a crime scene, identifying their type can be critical in linking suspects or events to specific locations. Using the **Glass Identification Dataset**, this project builds robust machine learning models to classify glass types based on their chemical composition.

---

## Dataset Summary

- **Source**: UCI Machine Learning Repository - Glass Identification Dataset
- **Samples**: 214
- **Features**: 9 numerical features (e.g., Refractive Index, Sodium, Magnesium, Aluminum, etc.)
- **Target Classes**: 6 classes (1, 2, 3, 5, 6, 7)  
  *(Class 4 is absent in this dataset)*

---

## Objectives

- Perform Exploratory Data Analysis (EDA)
- Handle class imbalance and visualize class distribution
- Train and evaluate the following models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Use **Cross-Validation** for model robustness
- Apply **Recursive Feature Elimination (RFE)** to rank feature importance
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction and 2D visualization
- Tune hyperparameters for Gradient Boosting and XGBoost
- Visualize results using **confusion matrices** and **classification reports**
- Deploy the best-performing model via **Streamlit**

---

## Exploratory Data Analysis (EDA)

- Analyzed class distribution and imbalance
- Checked pairwise feature correlation heatmaps
- Visualized feature distributions using histograms and boxplots
- Identified multicollinearity and potential outliers

---

## Model Development & Evaluation

### Models Trained:
| Model              | Technique                 |
|--------------------|---------------------------|
| K-Nearest Neighbors| Distance-based classifier |
| Decision Tree      | Tree-based model          |
| Random Forest      | Ensemble of decision trees|
| Gradient Boosting  | Boosted tree ensemble     |
| XGBoost            | Extreme Gradient Boosting |

### Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix (per class)
- Cross-validation accuracy
- Classification reports

---

## Feature Engineering

- **Recursive Feature Elimination (RFE)** applied to identify the most predictive features
- **PCA** used to reduce dimensions for 2D visualization of class separability

---

## Hyperparameter Tuning

Hyperparameter optimization was performed using `GridSearchCV` and `RandomizedSearchCV` on:

- **Gradient Boosting**
- **XGBoost**

Metrics used: Cross-validated accuracy and macro F1-score

---

## PCA Visualization

PCA transformed the dataset into 2 components to allow for 2D visualization, illustrating how well-separated the glass classes are in reduced dimensions.

---

## Deployment (Streamlit)

The final trained model (best-performing) was deployed using **Streamlit**, offering an interactive web application where forensic teams can:

- Input new glass composition
- Receive predicted glass type in real time

### Run Streamlit App Locally:

```bash
streamlit run app.py
````

---

## Credits

* Dataset: UCI Machine Learning Repository
* Libraries: scikit-learn, pandas, seaborn, matplotlib, xgboost, streamlit

---

## License

This project is open-source and available for educational or research use.

---

## Author

**Benedict Ouma**