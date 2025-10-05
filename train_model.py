# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

# Загрузить данные
df = pd.read_csv("air_quality_data.csv")

print(f"✅ Loaded {len(df)} rows")
print(df.head())

# Подготовить данные
feature_columns = [
    'AvgTemperature_C',
    'AvgWindSpeed_m_s',
    'AvgPrecipitation_mm',
    'CO_ppm',
    'NO2_ppb',
    'O3_ppb',
    'TrafficIndex'
]

X = df[feature_columns]
y = df['AQI']

# Разделить данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабировать
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучить модель
print("\n⏳ Training model...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Оценить качество
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"✅ Training R²: {train_score:.4f}")
print(f"✅ Testing R²: {test_score:.4f}")

# Сохранить модель
joblib.dump(model, "aqi_model.pkl")
joblib.dump(scaler, "aqi_scaler.pkl")

# Сохранить веса
weights = {}
for name, importance in zip(feature_columns, model.feature_importances_):
    weights[name] = float(importance)

# Нормализовать веса
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

with open("aqi_weights.json", 'w') as f:
    json.dump({
        'weights': weights,
        'feature_names': feature_columns,
        'metrics': {
            'train_r2': train_score,
            'test_r2': test_score
        }
    }, f, indent=2)

print("\n💾 Saved:")
print("  • aqi_model.pkl")
print("  • aqi_scaler.pkl")
print("  • aqi_weights.json")

print("\n📊 Learned weights:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:30s}: {weight:.4f}")