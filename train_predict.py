import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):

    # For Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler())
    ])

    # For Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("numerical", num_pipeline, num_attributes),
        ("categorical", cat_pipeline, cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Let's train the model.
    housing = pd.read_excel("housing.xlsx")

    # Create a stratified test set based on income category
    housing["income_category"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)

    for train_index, test_index in split.split(housing, housing['income_category']):
        housing.loc[test_index].drop('income_category', axis = 1).to_excel("input.xlsx", index=False)
        housing = housing.loc[train_index].drop('income_category', axis = 1) 

    # Separate predictors and labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis = 1)

    # List the numerical and categorical columns
    num_attributes = housing_features.drop("ocean_proximity", axis = 1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    Pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = Pipeline.fit_transform(housing_features)

    # Train the model
    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(Pipeline, PIPELINE_FILE)

    print("The model and pipeline have been trained and saved.")

else: 
    # Let's do inference using the saved model and pipeline
    model = joblib.load(MODEL_FILE)
    Pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_excel("input.xlsx")
    transformed_input =  Pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_excel("predictions.xlsx", index=False)
    print("Predictions have been saved to predictions.xlsx.")