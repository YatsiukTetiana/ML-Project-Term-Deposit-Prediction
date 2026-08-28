# ML-Project-Term-Deposit-Prediction  

Project is dedicated to create a binary classification model predicting the bank client's deposit agreement. The data contains direct marketing campaigns records (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

## Summary
### Project stages
1. **EDA**
   Contains exploratory data analysis (numerical and categorical data types, target value distribution, correlations) and initial hypothesis on deposit approval conditions
2. **Methodology**
   Description and explanation of chosen methods for data preprocessing, model types, main and additional model performance quality metrics. For *education* two options of encoding have to be compared: onehot vs ordinal.
3. **Data Preprocessing Module**
   Creating data preprocessing pipeline and functions, and refactoring them into python module
4. **Model Training and Performance Comparison Module**
   Creating functions for initiating, training, evaluating, and comparing models as custom python module functions
5. **Model Comparison**
   Hyperparameter optimization for models chosen in Methodology (RandomSearchCV, GridSearchCV, Cycle (for Decision Trees), Hyperopt):
   - *Logistic Regression*
   - *Decision Tree*
   - *kNN*
   - *AdaBoost*
   - *XGBoost*

   Based on main and additional model quality metrics chosen in Methodology, chose model with highest scores from each model type. Compared chosen models on validation subset, defined final model with the highest main quality score (AUROC) - XGBoost model with optimized hyperparameters from RandomSearchCV and ordinal education encoding.

   Quality metrics for each model type:
   - *Logistic Regression*
     The highest Recall values
     <img width="1111" height="130" alt="image" src="https://github.com/user-attachments/assets/7375db73-49cd-4378-91f4-afc13677dd9b" />

   - *Decision Tree*
     The highest F1-Score
     <img width="1016" height="131" alt="image" src="https://github.com/user-attachments/assets/09767688-db38-4856-93e7-323545337a14" />

   - *kNN*
     Possible overfit
     <img width="926" height="126" alt="image" src="https://github.com/user-attachments/assets/dac62a41-f9d9-4250-92f8-6f98c2292bf7" />

   - *AdaBoost*
     2nd highest AUROC, Precision, and Average Precision
     <img width="1020" height="127" alt="image" src="https://github.com/user-attachments/assets/588711a1-6b17-42a2-958b-e3c2347ce4c0" />

   - *XGBoost*
     The highest AUROC, Precision, and Average Precision. Is recommended for being chosen.
     <img width="1007" height="130" alt="image" src="https://github.com/user-attachments/assets/09ce5e21-e8ae-4ba5-b53f-ca93919396a3" />

6. **Feature Importance Analysis**
   For chosen XGBoost model, analyzed feature importance with built-in method.
   Top features by influence on target value are economical metrics, previous contact with a client, and success in previous campaign.

7. **Feature Influence on Model Prediction Analysis**
   Analyzed and displayed the direction and strenght of features influence on model predictions utilizing SHAP library.
   High negative influence: *nr.employed*, *emp.var.rate*, *cons.conf.idx*, *euribor3m*, day of call - Monday, high volume of calls during current campaign.
   High positive influence: celluar telephone type of contact, contacting client dureng previous campaigns

8. **Model Prediction Errors Analysis**
   Analyzed model errors rate with different thresholds, suggested to choose threshold = 0.25 for model predictions. Final decision on threshold depends on cost of failures/success for the company.
   In case when after choosing specific threshold model results are not quality enough, suggested either to change the main quality metric and model with highest value of this metric or develop XGBoost model by improving hyperparameters and feature importances.

   
# Project Files
- Full project Jupyter Notebook: [Mid_term_Project.ipynb]()  

Custom python modules:
- Preprocessign original dataset, splitting to train, test, and validation subsets: [process_bank_deposit.py]()
- Creating, training, and getting models predictions: [model_evaluation.py](https://github.com/YatsiukTetiana/ML-Project-Term-Deposit-Prediction/blob/main/model_evaluation.py)
- Comparing models preformance by key metrics: [models_comparison.py](https://github.com/YatsiukTetiana/ML-Project-Term-Deposit-Prediction/blob/main/models_comparison.py)
