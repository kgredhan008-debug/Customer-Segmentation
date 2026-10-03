# Customer Segmentation – Phase 4

## Dataset
Customer Segmentation dataset: 10,695 records and 12 columns.

## Final Model
Random Forest Classifier with:
- 80/20 stratified train-test split
- Median imputation for numeric missing values
- Most-frequent imputation for categorical missing values
- One-hot encoding for categorical features
- Standardization of numeric features
- 300 trees, random_state=42, balanced class weights

## Result
Test accuracy: 40.30%

## Features
Gender, Ever_Married, Age, Graduated, Profession, Work_Experience,
Spending_Score, Family_Size and Var_1.

ID and Unnamed: 0 are excluded because they are identifiers/index values.

## Clustering diagnostic
K-Means silhouette analysis was also performed for k=2..8.
Best silhouette score: 0.1946 at k=3.
This is a diagnostic for unsupervised clustering and is separate from the final supervised classifier.

## Run the application
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- The classification accuracy is moderate.
- Dataset labels may overlap in feature space.
- Results depend on the supplied dataset and preprocessing.
- No external customer behavior data is included.

## Future Enhancements
- Hyperparameter tuning and cross-validation.
- Try Gradient Boosting/XGBoost-style models.
- Add probability/confidence display.
- Add interactive EDA and segment charts.
- Deploy the Streamlit app to a cloud service.

## GitHub
Create a repository and upload `app.py`, `customer_segmentation_model.pkl`,
`requirements.txt`, `README.md`, and the project report. Do not upload
sensitive/private customer data.
