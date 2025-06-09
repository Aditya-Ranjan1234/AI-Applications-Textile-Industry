# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# import pandas as pd

# # Load the dataset
# df = pd.read_csv("synthetic_garment_demand.csv")

# # Define features and target
# target = 'demand'
# features = df.columns.drop(['demand', 'date'])

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# # Define categorical and numeric columns
# categorical_cols = ['day_of_week', 'category', 'product_id', 'gender', 'size', 'color', 'location']
# numeric_cols = list(set(features) - set(categorical_cols))

# # Preprocessing pipelines
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean'))
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Column transformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

# # Full pipeline with model
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
# ])

# # Train the model
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluate
# mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(mae)
# print(rmse)
# print(r2)
# mae, rmse, r2

# def predict_demand(feature_array):
#     """
#     Predicts garment demand given a list of feature dictionaries.

#     Parameters:
#     - feature_array: List[Dict], each dict contains keys matching model's features
#                      (excluding 'demand' and 'date').

#     Returns:
#     - numpy array of predicted demand values.
#     """
#     # Convert the list of dicts into a DataFrame
#     input_df = pd.DataFrame(feature_array)

#     # Use the trained pipeline to make predictions
#     predictions = model.predict(input_df)

#     return predictions
# sample_input = [{
#     "day_of_week": "Monday",
#     "category": "T-shirt",
#     "product_id": "P1001",
#     "gender": "Men",
#     "size": "M",
#     "color": "Blue",
#     "price": 499,
#     "discount": 10,
#     "stock_available": 200,
#     "holiday_flag": 0,
#     "event_flag": 0,
#     "temperature": 32.5,
#     "rainfall": 0.3,
#     "location": "Mumbai"
# }]

# print(predict_demand(sample_input))  # e.g. [102.41]


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("synthetic_garment_demand.csv")

# Define features and target
target = 'demand'
features = df.columns.drop(['demand', 'date'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Define categorical and numeric columns
categorical_cols = ['day_of_week', 'category', 'product_id', 'gender', 'size', 'color', 'location']
numeric_cols = list(set(features) - set(categorical_cols))

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Full pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Streamlit App
st.title("👕 Garment Demand Predictor")

st.markdown("Enter the product details below to predict demand:")

# User inputs
day_of_week = st.selectbox("Day of Week", df["day_of_week"].unique())
category = st.selectbox("Category", df["category"].unique())
product_id = st.selectbox("Product ID", df["product_id"].unique())
gender = st.selectbox("Gender", df["gender"].unique())
size = st.selectbox("Size", df["size"].unique())
color = st.selectbox("Color", df["color"].unique())
price = st.number_input("Price (INR)", min_value=100, max_value=5000, value=499)
discount = st.slider("Discount (%)", 0, 50, 10)
stock_available = st.slider("Stock Available", 0, 500, 200)
holiday_flag = st.radio("Is it a holiday?", [0, 1])
event_flag = st.radio("Is there an event?", [0, 1])
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 30.0)
rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 0.3)
location = st.selectbox("Location", df["location"].unique())

# Predict button
if st.button("Predict Demand"):
    sample_input = [{
        "day_of_week": day_of_week,
        "category": category,
        "product_id": product_id,
        "gender": gender,
        "size": size,
        "color": color,
        "price": price,
        "discount": discount,
        "stock_available": stock_available,
        "holiday_flag": holiday_flag,
        "event_flag": event_flag,
        "temperature": temperature,
        "rainfall": rainfall,
        "location": location
    }]
    
    prediction = model.predict(pd.DataFrame(sample_input))[0]
    
    st.success(f"📦 Estimated Demand: **{int(prediction)} units**")
