# ============================================
# train_model.py — Re-train AQI Prediction Model (V3)
# ============================================

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. LOAD DATASET
# ============================================

DATA_PATH = "dataset.csv"

print("============================================================")
print("📂 LOADING DATASET")
print("============================================================")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {DATA_PATH}")
print(f"Shape: {df.shape}")

# ============================================
# 2. CLEAN & ENCODE DATA
# ============================================

# Drop non-numeric columns (like city names)
non_numeric = [col for col in df.columns if df[col].dtype == 'object']
if non_numeric:
    print(f"⚠️ Dropping non-numeric columns: {non_numeric}")
    df = df.drop(columns=non_numeric)

# Drop rows with missing AQI
df = df.dropna(subset=["AQI"])

# Fill missing values with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  • Filled {col} missing values with median: {round(median_val,2)}")

# ============================================
# 3. ADD ENGINEERED FEATURES
# ============================================

print("============================================================")
print("🧩 FEATURE ENGINEERING")
print("============================================================")

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    df = pd.get_dummies(df, columns=["Season"], drop_first=True)
    print("✅ Added temporal and season features")

# Add interactions
if all(col in df.columns for col in ["CO_ppm", "TrafficIndex"]):
    df["CO_Traffic_Interaction"] = df["CO_ppm"] * df["TrafficIndex"]

if all(col in df.columns for col in ["NO2_ppb", "TrafficIndex"]):
    df["NO2_Traffic_Interaction"] = df["NO2_ppb"] * df["TrafficIndex"]

if all(col in df.columns for col in ["O3_ppb", "AvgTemperature_C"]):
    df["O3_Temp_Interaction"] = df["O3_ppb"] * df["AvgTemperature_C"]

if all(col in df.columns for col in ["AvgWindSpeed_m_s", "TrafficIndex"]):
    df["WindDispersion"] = df["AvgWindSpeed_m_s"] / (df["TrafficIndex"] + 1)

print(f"✅ Added engineered features")
print(f"Total columns now: {len(df.columns)}")

# ============================================
# 4. DEFINE FEATURES AND TARGET
# ============================================

target = "AQI"
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

# ============================================
# 5. TRAIN/TEST SPLIT + SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================
# 6. TRAIN MODEL
# ============================================

print("\n============================================================")
print("🎯 TRAINING MODEL")
print("============================================================")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=12,
    min_samples_split=5,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================
# 7. EVALUATION
# ============================================

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

cv_rmse = np.mean(np.sqrt(-cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_squared_error")))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"CV RMSE (5-fold): {cv_rmse:.2f}")

if train_r2 - test_r2 > 0.25:
    print("⚠️ Warning: Possible overfitting detected!")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================

importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df["normalized"] = importance_df["importance"] / importance_df["importance"].sum()

print("\n============================================================")
print("📊 LEARNED WEIGHTS (Wi)")
print("============================================================")

for i, row in enumerate(importance_df.itertuples(), 1):
    bar = "█" * int(row.normalized * 30)
    print(f"{i:2d}. {row.feature:30s} | {row.normalized:.4f} | {bar}")

print("============================================================")
print(f"Total: {importance_df['normalized'].sum():.4f}")
print("============================================================")

# ============================================
# 9. SAVE MODEL, SCALER, WEIGHTS
# ============================================

joblib.dump(model, "aqi_model.pkl")
joblib.dump(scaler, "aqi_scaler.pkl")

weights_dict = {
    "weights": dict(zip(importance_df["feature"], importance_df["normalized"]))
}
with open("aqi_weights.json", "w") as f:
    json.dump(weights_dict, f, indent=4)

print("\n💾 Saved:")
print("  • aqi_model.pkl")
print("  • aqi_scaler.pkl")
print("  • aqi_weights.json")

# ============================================
# 10. TEST PREDICTION (DEMO)
# ============================================

print("\n============================================================")
print("🔮 MAKING PREDICTIONS")
print("============================================================")

sample = X.iloc[0].to_dict()
X_demo = scaler.transform([list(sample.values())])
pred_aqi = model.predict(X_demo)[0]

pollution_index = round(np.sum(importance_df["normalized"]) * 100, 2)
top_factors = importance_df.head(5)

print(f"📊 Pollution Index: {pollution_index}")
print(f"📈 Predicted AQI: {pred_aqi:.2f}\n")

print("🔝 Top 5 Contributing Factors:")
for i, row in enumerate(top_factors.itertuples(), 1):
    print(f"  {i}. {row.feature}: {row.normalized * 100:.2f}% (weight: {row.normalized:.4f})")

print("\n✅ Training complete!\n")
