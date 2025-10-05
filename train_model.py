# train_model.py
# Обучение модели на вашем датасете с правильными колонками

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os

def train_aqi_model(dataset_path="dataset.csv"):
    """
    Обучение модели на датасете с колонками:
    City, Date, AvgTemperature_C, AvgWindSpeed_m_s, AvgPrecipitation_mm,
    CO_ppm, NO2_ppb, O3_ppb, AQI, TrafficIndex
    """
    
    print("="*60)
    print("🚀 TRAINING AQI PREDICTION MODEL")
    print("="*60)
    
    # 1. Загрузка данных
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"\n✅ Loaded {len(df)} rows from {dataset_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst rows:\n{df.head()}")
    
    # 2. Проверка наличия необходимых колонок
    required_cols = [
        'AvgTemperature_C', 'AvgWindSpeed_m_s', 'AvgPrecipitation_mm',
        'CO_ppm', 'NO2_ppb', 'O3_ppb', 'TrafficIndex', 'AQI'
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # 3. Обработка пропусков
    print(f"\n📊 Data info:")
    print(f"  • Missing values:\n{df[required_cols].isnull().sum()}")
    
    # Удаляем строки с пропусками в целевой переменной
    df = df.dropna(subset=['AQI'])
    
    # Заполняем пропуски в признаках медианами
    for col in required_cols:
        if col != 'AQI' and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  • Filled {col} missing values with median: {median_val:.2f}")
    
    print(f"\n✅ Clean dataset: {len(df)} rows")
    
    # 4. Подготовка данных
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
    
    print(f"\n📊 Target variable (AQI) statistics:")
    print(f"  • Mean: {y.mean():.2f}")
    print(f"  • Median: {y.median():.2f}")
    print(f"  • Std: {y.std():.2f}")
    print(f"  • Min: {y.min():.2f}")
    print(f"  • Max: {y.max():.2f}")
    
    # 5. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✅ Split data:")
    print(f"  • Training: {len(X_train)} samples")
    print(f"  • Testing: {len(X_test)} samples")
    
    # 6. Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n⏳ Scaling features...")
    
    # 7. Обучение модели
    print("\n⏳ Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 8. Оценка качества
    print("\n📈 Evaluating model...")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n✅ MODEL PERFORMANCE:")
    print("="*60)
    print(f"Training Set:")
    print(f"  • R² Score: {train_r2:.4f}")
    print(f"  • RMSE: {train_rmse:.2f}")
    print(f"  • MAE: {train_mae:.2f}")
    print(f"\nTest Set:")
    print(f"  • R² Score: {test_r2:.4f}")
    print(f"  • RMSE: {test_rmse:.2f}")
    print(f"  • MAE: {test_mae:.2f}")
    print("="*60)
    
    # 9. Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 FEATURE IMPORTANCE:")
    print("="*60)
    for feat, imp in sorted_importance:
        print(f"  {feat:30s}: {imp:.4f} {'█' * int(imp * 100)}")
    print("="*60)
    
    # 10. Сохранение модели
    print("\n💾 Saving model...")
    
    joblib.dump(model, "aqi_model.pkl")
    joblib.dump(scaler, "aqi_scaler.pkl")
    
    # 11. Сохранение весов (нормализованных)
    total_importance = sum(feature_importance.values())
    normalized_weights = {k: v/total_importance for k, v in feature_importance.items()}
    
    weights_data = {
        'weights': normalized_weights,
        'feature_names': feature_columns,
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'n_samples': len(df),
            'trained_at': pd.Timestamp.now().isoformat()
        },
        'model_params': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    }
    
    with open("aqi_weights.json", 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print("\n✅ SAVED FILES:")
    print("  • aqi_model.pkl")
    print("  • aqi_scaler.pkl")
    print("  • aqi_weights.json")
    
    # 12. Тестовое предсказание
    print("\n🧪 TEST PREDICTION:")
    print("="*60)
    
    test_sample = X_test.iloc[0:1]
    test_actual = y_test.iloc[0]
    test_scaled = scaler.transform(test_sample)
    test_pred = model.predict(test_scaled)[0]
    
    print(f"Input features:")
    for col in feature_columns:
        print(f"  • {col}: {test_sample[col].values[0]:.2f}")
    print(f"\nActual AQI: {test_actual:.2f}")
    print(f"Predicted AQI: {test_pred:.2f}")
    print(f"Error: {abs(test_actual - test_pred):.2f}")
    print("="*60)
    
    print("\n✅ TRAINING COMPLETE!")
    print("\nYou can now run the API server with: python app/main.py")
    
    return model, scaler, weights_data


if __name__ == "__main__":
    try:
        train_aqi_model("dataset.csv")
    except FileNotFoundError:
        print("\n❌ ERROR: dataset.csv not found!")
        print("\nPlease ensure your dataset has these columns:")
        print("  - City")
        print("  - Date")
        print("  - AvgTemperature_C")
        print("  - AvgWindSpeed_m_s")
        print("  - AvgPrecipitation_mm")
        print("  - CO_ppm")
        print("  - NO2_ppb")
        print("  - O3_ppb")
        print("  - AQI")
        print("  - TrafficIndex")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()