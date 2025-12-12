import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
housing = pd.read_excel("housing.xlsx")

# 2. Create a stratified test set based on income category
housing["income_category"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)
for train_index, test_index in split.split(housing, housing['income_category']):
    strat_train_set = housing.loc[train_index].drop('income_category', axis = 1)  # we will work on this data.
    strat_test_set = housing.loc[test_index].drop('income_category', axis = 1)   # Set aside for testing.

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis = 1)

# print(housing, housing_labels)

# 4. List the numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis = 1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# 5. Let's make the Pipeline

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

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
 
# housing_prepared is now a NumPy array ready for training
# print(housing_prepared)  

# 7. Train various models

# # Linear Regression Model:-
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_reg_predictions = lin_reg.predict(housing_prepared)

lin_reg_rmse = root_mean_squared_error(housing_labels, lin_reg_predictions)
lin_reg_rmses_cv = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

print(f"The root mean squared error(RMSE) for Linear Regression is: {lin_reg_rmse}")
print(f"the Cross-Validated RMSE scores for Linear Regression are: {pd.Series(lin_reg_rmses_cv).describe()}")


# Decision Tree Model:-
"""
⚠️ Note:
:- If the RMSE value for DecisionTreeRegressor is 0, it usually means the model was evaluated on the same data it was trained on.
Decision trees can perfectly memorize (overfit) the training data, resulting in zero error on that data — but it won't perform well on new/unseen data.
Always use a train-test split or cross-validation(cv) to get a realistic RMSE value.

:- A low CV RMSE is good — it means your model generalizes well —
but make sure it’s not much lower on training data and that you’re not leaking information into training.

:- Here, we are using RandomForestRegressor due to it occurs less error.
"""
dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(housing_prepared, housing_labels)
dec_tree_reg_predictions = dec_tree_reg.predict(housing_prepared)

dec_tree_reg_rmse = root_mean_squared_error(housing_labels, dec_tree_reg_predictions)
dec_tree_reg_rmses_cv = -cross_val_score(dec_tree_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

print(f"The root mean squared error(RMSE) for Decision Tree Regression is: {dec_tree_reg_rmse}")
print(f"the Cross-Validated RMSE scores for Decision Tree Regression are: {pd.Series(dec_tree_reg_rmses_cv).describe()}")

# Random Forest Model:-
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_reg_predictions = random_forest_reg.predict(housing_prepared)

random_forest_reg_rmse = root_mean_squared_error(housing_labels, random_forest_reg_predictions)
random_forest_reg_rmses_cv = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

print(f"The root mean squared error(RMSE) for Random Forest Regression is: {random_forest_reg_rmse}")
print(f"the Cross-Validated RMSE scores for Random Forest Regression are: {pd.Series(random_forest_reg_rmses_cv).describe()}")