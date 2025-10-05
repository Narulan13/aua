"""
Конвертация единиц измерения загрязнителей
Приводим все к стандартным единицам для модели
"""

def convert_to_standard_units(pollutant: str, value: float, unit: str) -> float:
    """
    Конвертирует значение загрязнителя в стандартные единицы
    
    Стандартные единицы для модели:
    - PM2.5, PM10: μg/m³
    - NO2: ppb (parts per billion)
    - O3: ppb
    - CO: ppm (parts per million)
    - SO2: ppb
    
    Args:
        pollutant: название загрязнителя ('pm25', 'no2', etc.)
        value: значение
        unit: текущая единица измерения
        
    Returns:
        Значение в стандартных единицах
    """
    
    pollutant = pollutant.lower()
    unit = unit.lower().replace(' ', '')
    
    # PM2.5 и PM10 (стандарт: μg/m³)
    if pollutant in ['pm25', 'pm2.5', 'pm10']:
        if 'μg/m³' in unit or 'ug/m3' in unit or 'µg/m³' in unit:
            return value
        elif 'mg/m³' in unit or 'mg/m3' in unit:
            return value * 1000  # mg/m³ → μg/m³
        else:
            return value  # Assume already in μg/m³
    
    # NO2 (стандарт: ppb)
    elif pollutant == 'no2':
        if 'ppb' in unit:
            return value
        elif 'ppm' in unit:
            return value * 1000  # ppm → ppb
        elif 'μg/m³' in unit or 'ug/m3' in unit or 'µg/m³' in unit:
            # NO2: 1 ppb = 1.88 μg/m³ at 25°C
            return value / 1.88  # μg/m³ → ppb
        elif 'mol/m²' in unit:
            # TEMPO NO2 column (tropospheric)
            # Примерная конвертация: 1e15 molecules/cm² ≈ 10-50 ppb приземного NO2
            # Это грубая оценка для демо
            return value * 1e15 * 30  # Упрощенная формула
        else:
            return value
    
    # O3 (стандарт: ppb)
    elif pollutant == 'o3':
        if 'ppb' in unit:
            return value
        elif 'ppm' in unit:
            return value * 1000  # ppm → ppb
        elif 'μg/m³' in unit or 'ug/m3' in unit or 'µg/m³' in unit:
            # O3: 1 ppb = 1.96 μg/m³ at 25°C
            return value / 1.96  # μg/m³ → ppb
        else:
            return value
    
    # CO (стандарт: ppm)
    elif pollutant == 'co':
        if 'ppm' in unit:
            return value
        elif 'ppb' in unit:
            return value / 1000  # ppb → ppm
        elif 'μg/m³' in unit or 'ug/m3' in unit or 'µg/m³' in unit:
            # CO: 1 ppm = 1145 μg/m³ at 25°C
            return value / 1145  # μg/m³ → ppm
        elif 'mg/m³' in unit or 'mg/m3' in unit:
            return value / 1.145  # mg/m³ → ppm
        else:
            return value
    
    # SO2 (стандарт: ppb)
    elif pollutant == 'so2':
        if 'ppb' in unit:
            return value
        elif 'ppm' in unit:
            return value * 1000  # ppm → ppb
        elif 'μg/m³' in unit or 'ug/m3' in unit or 'µg/m³' in unit:
            # SO2: 1 ppb = 2.62 μg/m³ at 25°C
            return value / 2.62  # μg/m³ → ppb
        else:
            return value
    
    # Если не распознали pollutant или unit, возвращаем как есть
    return value


def aqi_from_pm25(pm25: float) -> int:
    """
    Вычисляет AQI из концентрации PM2.5 (EPA стандарт)
    
    Args:
        pm25: концентрация PM2.5 в μg/m³
        
    Returns:
        AQI значение (0-500)
    """
    if pm25 <= 12.0:
        return int((50 / 12.0) * pm25)
    elif pm25 <= 35.4:
        return int(50 + ((100 - 50) / (35.4 - 12.0)) * (pm25 - 12.0))
    elif pm25 <= 55.4:
        return int(100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4))
    elif pm25 <= 150.4:
        return int(150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4))
    elif pm25 <= 250.4:
        return int(200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4))
    else:
        return min(500, int(300 + ((500 - 300) / (500.4 - 250.4)) * (pm25 - 250.4)))


def get_aqi_category(aqi: int) -> dict:
    """
    Возвращает категорию AQI и рекомендации
    
    Args:
        aqi: AQI значение
        
    Returns:
        Dict с категорией, цветом и описанием
    """
    if aqi <= 50:
        return {
            'category': 'Good',
            'color': '#00e400',
            'description': 'Air quality is satisfactory, and air pollution poses little or no risk.'
        }
    elif aqi <= 100:
        return {
            'category': 'Moderate',
            'color': '#ffff00',
            'description': 'Air quality is acceptable. However, there may be a risk for some people.'
        }
    elif aqi <= 150:
        return {
            'category': 'Unhealthy for Sensitive Groups',
            'color': '#ff7e00',
            'description': 'Members of sensitive groups may experience health effects.'
        }
    elif aqi <= 200:
        return {
            'category': 'Unhealthy',
            'color': '#ff0000',
            'description': 'Some members of the general public may experience health effects.'
        }
    elif aqi <= 300:
        return {
            'category': 'Very Unhealthy',
            'color': '#8f3f97',
            'description': 'Health alert: The risk of health effects is increased for everyone.'
        }
    else:
        return {
            'category': 'Hazardous',
            'color': '#7e0023',
            'description': 'Health warning of emergency conditions: everyone is more likely to be affected.'
        }


# Примеры использования
if __name__ == "__main__":
    print("Testing unit conversions:\n")
    
    # Test PM2.5
    print("PM2.5:")
    print(f"  35 μg/m³ = {convert_to_standard_units('pm25', 35, 'μg/m³')} μg/m³")
    print(f"  0.035 mg/m³ = {convert_to_standard_units('pm25', 0.035, 'mg/m³')} μg/m³")
    
    # Test NO2
    print("\nNO2:")
    print(f"  50 ppb = {convert_to_standard_units('no2', 50, 'ppb')} ppb")
    print(f"  0.05 ppm = {convert_to_standard_units('no2', 0.05, 'ppm')} ppb")
    print(f"  94 μg/m³ = {convert_to_standard_units('no2', 94, 'μg/m³'):.1f} ppb")
    
    # Test O3
    print("\nO3:")
    print(f"  70 ppb = {convert_to_standard_units('o3', 70, 'ppb')} ppb")
    print(f"  137.2 μg/m³ = {convert_to_standard_units('o3', 137.2, 'μg/m³'):.1f} ppb")
    
    # Test CO
    print("\nCO:")
    print(f"  1.5 ppm = {convert_to_standard_units('co', 1.5, 'ppm')} ppm")
    print(f"  1717.5 μg/m³ = {convert_to_standard_units('co', 1717.5, 'μg/m³'):.1f} ppm")
    
    # Test AQI
    print("\n\nAQI Calculations:")
    for pm25 in [10, 25, 45, 75, 200]:
        aqi = aqi_from_pm25(pm25)
        category = get_aqi_category(aqi)
        print(f"  PM2.5 {pm25} μg/m³ → AQI {aqi} ({category['category']})")