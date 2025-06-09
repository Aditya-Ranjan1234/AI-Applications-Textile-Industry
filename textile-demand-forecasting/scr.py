# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load your dataset
df = pd.read_csv("Textile_sales_data.csv")

# 🔹 Parse the DATE column flexibly
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce', dayfirst=True)

# 🔹 Drop invalid or missing dates
df = df.dropna(subset=['DATE'])

# 🔹 Extract useful date parts
df['Month'] = df['DATE'].dt.month
df['Year'] = df['DATE'].dt.year

# 🔹 Define features and target
features = ['VALUE', 'PFco_Code', 'Total_values', 'Month', 'Year']
X = df[features]
y = df['QUANTITIES_Kgs']

# 🔹 Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate performance
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# 🔹 Feature importance visualization
importances = model.feature_importances_
feature_names = features
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# 🔹 Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - XGBoost Regression')
plt.tight_layout()
plt.show()
