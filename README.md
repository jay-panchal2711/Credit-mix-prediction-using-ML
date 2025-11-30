<img width="905" height="517" alt="Screenshot 2025-11-30 at 11 42 36 PM" src="https://github.com/user-attachments/assets/5ade8d58-fe6a-42d0-8630-65df83ef235b" /># Predictive Modeling Using Classification Algorithms
### Credit Mix Prediction | End-to-End Machine Learning Pipeline

---

## 📌 Project Overview
This project focuses on building an end-to-end Machine Learning classification system to predict **Credit Mix**, an important financial risk indicator that represents the combination of credit products that a customer utilizes (Credit cards, loans, mortgages, etc.).

The primary goal is to analyze financial attributes, process the dataset, apply classification algorithms, evaluate performance, and determine the most efficient model for predicting credit mix categories.

---

## 🎯 Objective
To develop a robust ML classification model that accurately predicts **Credit Mix** categories based on an individual's financial and behavioral features.

---

## 🧾 Target Column

Possible classes:
- **0 → Standard**
- **1 → Poor**
- **2 → Good**
- **3 → Excellent** *(if applicable)*

---

## 📂 Dataset Description
The dataset contains customer-level data including demographic and financial attributes such as:
- Annual Income
- Monthly Salary
- Number of Loans
- Credit Card Usage
- Outstanding Debt
- Credit Utilization Ratio
- Number of Delayed Payments
- Total EMI per Month

### **Initial Data Checks**
- Shape and structure analysis
- Missing data treatment
- Duplicate removal
- Outlier visualization
- Correlation study

---

## 🧠 Steps Followed
1. Data loading using Pandas
2. Exploratory data analysis (EDA)
3. Feature engineering & preprocessing
4. Train-Test Split (80:20)
5. Label Encoding & Standard Scaling
6. Model training with ML classification algorithms
7. Hyperparameter tuning (Random Search / Grid Search for RF & XGBoost)
8. Evaluation using accuracy, precision, recall, F1 score, confusion matrix
9. Final comparison and selection of the best model

---

## 🤖 Models Trained
| Model | Accuracy |
|--------|----------|
| **Random Forest (Best)** | **0.7712** |
| XGBoost | 0.7633 |
| Logistic Regression | 0.6971 |
| Naive Bayes | 0.6914 |
| Decision Tree | 0.6584 |
| AdaBoost | 0.6520 |

---

## 📊 Model Performance (Random Forest Tuned)
Test Accuracy: 0.7707

Precision (weighted): 0.6765

Recall (weighted): 0.7707

F1 Score (weighted): 0.7035



### **Classification Report**
| Class | Precision | Recall | F1-score |
|--------|-----------|----------|------------|
| 0 | 0.80 | 0.94 | 0.87 |
| 1 | 0.22 | 0.03 | 0.05 |
| 2 | 0.78 | 0.95 | 0.85 |
| 3 | 0.77 | 0.95 | 0.85 |

---

## 📌 Confusion Matrix (Tuned Random Forest)

*<img width="497" height="600" alt="Screenshot 2025-11-30 at 10 53 28 PM" src="https://github.com/user-attachments/assets/498b2a08-7468-473e-b130-1c135af23d6f" />
*

---

## 📌 Model Comparison Table

<img width="488" height="239" alt="s2" src="https://github.com/user-attachments/assets/fdcbc4b4-aad3-4b7f-b2d2-6c0578313043" />


---

## 📌 Correlation Heatmap

<img width="905" height="517" alt="Screenshot 2025-11-30 at 11 42 36 PM" src="https://github.com/user-attachments/assets/17e13f4a-2433-4b93-ac5a-561a52f7cb9b" />

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🧾 Conclusion
- The **Random Forest Classifier** gave the best performance with an accuracy of **~77%**
- High accuracy on Class 2 & Class 3 categories
- Model still struggles with minority class (Class 1 under-represented)
- Further tuning & resampling techniques (SMOTE, ADASYN) may improve performance

---

## ⚠ Drawbacks / Limitations
- Class imbalance impacts prediction results
- More advanced models such as LightGBM or CatBoost could be tested
- No cloud or real-time deployment implemented yet
- Dataset lacks additional behavioral data that could improve accuracy

---

## 🚀 Future Improvements
- Apply SMOTE balancing technique
- Perform hyperparameter tuning using GridSearchCV
- Deploy model using Flask / FastAPI / Streamlit
- Integrate SHAP for interpretability

---



---

## 👤 Author
**Jay Panchal**  
Machine Learning & Data Science Enthusiast  

📧 Email: **panchaljay2711@gmail.com**  
🔗 LinkedIn: *https://www.linkedin.com/in/jay-panchal-396443176/*  
🐙 GitHub: *https://github.com/jay-panchal2711*

---

⭐ *If you found this project helpful, please consider giving the repository a star!*

