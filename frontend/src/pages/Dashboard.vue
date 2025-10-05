<template>
  <!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirQualityAI - Прогноз качества воздуха</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            color: #333;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.1rem;
        }

        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }

        .tabs {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .tab-btn {
            flex: 1;
            padding: 1rem;
            border: none;
            background: #f5f5f5;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }

        .tab-btn.active {
            background: #667eea;
            color: white;
            font-weight: bold;
        }

        .tab-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .loading {
            background: white;
            padding: 3rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .card h3 {
            margin-bottom: 1rem;
            color: #333;
        }

        .card-full {
            grid-column: 1 / -1;
        }

        .profile-inputs {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }

        .profile-inputs label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.95rem;
        }

        .profile-inputs input[type="number"] {
            width: 80px;
            padding: 0.5rem;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1rem;
        }

        .profile-inputs input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .aqi-card {
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1rem;
            color: white;
        }

        .aqi-good { background: linear-gradient(135deg, #00e400, #00b300); }
        .aqi-moderate { background: linear-gradient(135deg, #ffff00, #ffd700); color: #333; }
        .aqi-unhealthy-sensitive { background: linear-gradient(135deg, #ff7e00, #ff5500); }
        .aqi-unhealthy { background: linear-gradient(135deg, #ff0000, #cc0000); }
        .aqi-very-unhealthy { background: linear-gradient(135deg, #8f3f97, #6a1b9a); }

        .aqi-card h2 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }

        .data-sources {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            font-size: 0.85rem;
        }

        .weights-display {
            margin-top: 1rem;
            padding: 1rem;
            background: #f9f9f9;
            border-radius: 8px;
        }

        .weight-bar {
            margin: 0.5rem 0;
        }

        .weight-bar-fill {
            height: 24px;
            background: #667eea;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 0.5rem;
            color: white;
            font-size: 0.85rem;
            transition: width 0.3s;
        }

        .advice-section ul {
            list-style: none;
            padding: 0;
        }

        .advice-section li {
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: #f5f5f5;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            font-size: 0.95rem;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #c62828;
        }

        .hidden {
            display: none;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: white;
            opacity: 0.8;
            margin-top: 2rem;
        }

        /* Forecast chart container responsiveness */
        #forecastChart {
            width: 100% !important;
            max-height: 400px;
        }

    </style>
</head>
<body>
    <header class="header">
        <h1>🌍 AirQualityAI</h1>
        <p class="subtitle">Персональный прогноз качества воздуха с ML и динамическими весами</p>
    </header>

    <div class="container">
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Определяем ваше местоположение...</p>
        </div>

        <div id="content" class="hidden">
            <!-- Tabs -->
            <div class="tabs">
                <button class="tab-btn active" id="currentTabBtn">Текущий прогноз</button>
                <button class="tab-btn" id="forecastTabBtn">Прогноз на неделю</button>
            </div>

            <!-- Current Tab -->
            <div id="currentTab" class="tab-content active">
                <div class="content">
                    <div class="card">
                        <h3>👤 Ваш профиль</h3>
                        <div class="profile-inputs">
                            <label>
                                Возраст:
                                <input type="number" id="age" value="30" min="1" max="120">
                            </label>
                            <label>
                                <input type="checkbox" id="asthma">
                                У меня астма
                            </label>
                        </div>
                        <button class="btn btn-primary" onclick="app.getPrediction()">
                            🔄 Обновить данные
                        </button>
                    </div>

                    <div class="card">
                        <div id="currentAqi"></div>
                    </div>

                    <div class="card card-full">
                        <h3>⚖️ Динамические веса факторов</h3>
                        <div id="weightsDisplay" class="weights-display"></div>
                    </div>

                    <div class="card card-full">
                        <h3>💡 Персональные рекомендации</h3>
                        <div class="advice-section">
                            <ul id="adviceList"></ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Forecast Tab -->
            <div id="forecastTab" class="tab-content">
                <div class="content">
                    <div class="card card-full">
                        <h3>📅 Прогноз AQI на неделю</h3>
                        <canvas id="forecastChart"></canvas>
                        <div style="display:flex; gap:1rem; margin-top:1rem;">
                            <button class="btn btn-primary" style="width:auto; padding:0.6rem 1rem;" onclick="app.getForecast()">🔄 Обновить прогноз</button>
                            <button class="btn" style="width:auto; padding:0.6rem 1rem;" onclick="app.downloadForecastCSV()">⬇️ Скачать CSV</button>
                        </div>
                        <div id="forecastInfo" style="margin-top:1rem; font-size:0.9rem; color:#444;"></div>
                    </div>

                    <div class="card card-full">
                        <h3>🌡️ Погодные метрики (темп., ветер, осадки)</h3>
                        <div id="weatherTable" style="overflow:auto; max-height:260px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <div id="error" class="error hidden"></div>
    </div>

    <footer class="footer">
        <p>AirQualityAI v3.0 | Real-time data with dynamic weights</p>
    </footer>

    <script>
        const app = {
            location: null,
            prediction: null,
            forecastData: null,
            backendUrl: 'http://localhost:8000',
            forecastChart: null,

            init() {
                this.setupTabs();
                this.requestLocation();
            },

            setupTabs() {
                const currentBtn = document.getElementById('currentTabBtn');
                const forecastBtn = document.getElementById('forecastTabBtn');
                currentBtn.addEventListener('click', () => this.showTab('currentTab'));
                forecastBtn.addEventListener('click', () => this.showTab('forecastTab'));
            },

            showTab(tabId) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                if (tabId === 'forecastTab') document.getElementById('forecastTabBtn').classList.add('active');
                else document.getElementById('currentTabBtn').classList.add('active');
                document.getElementById(tabId).classList.add('active');
            },

            requestLocation() {
                if (!navigator.geolocation) {
                    this.showError('Геолокация не поддерживается');
                    return;
                }

                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        this.location = {
                            lat: position.coords.latitude,
                            lon: position.coords.longitude
                        };
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('content').classList.remove('hidden');
                        this.getPrediction();
                    },
                    (err) => {
                        this.showError('Не удалось определить местоположение');
                        console.error(err);
                    }
                );
            },

            async getPrediction() {
                if (!this.location) return;

                const profile = {
                    age: parseInt(document.getElementById('age').value),
                    asthma: document.getElementById('asthma').checked
                };

                try {
                    const response = await fetch(`${this.backendUrl}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            lat: this.location.lat,
                            lon: this.location.lon,
                            profile: profile
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'API error');
                    }

                    this.prediction = await response.json();
                    console.log('Prediction:', this.prediction);
                    this.displayCurrentAQI();
                    this.displayWeights();
                    this.hideError();

                } catch (err) {
                    console.error('Prediction error:', err);
                    this.showError(`Ошибка получения данных: ${err.message}`);
                }
            },

            displayCurrentAQI() {
                if (!this.prediction) return;

                const aqi = this.prediction.predicted_aqi || 50;
                const pollutionIndex = this.prediction.pollution_index || 50;

                let pm25Display = 'N/A';
                if (this.prediction.air_quality_data && (this.prediction.air_quality_data.NO2_ppb || this.prediction.air_quality_data.PM25)) {
                    const airData = this.prediction.air_quality_data;
                    pm25Display = (airData.PM25 ? `PM2.5: ${airData.PM25} μg/m³, ` : '') +
                                  (airData.NO2_ppb ? `NO2: ${airData.NO2_ppb.toFixed(1)} ppb, ` : '') +
                                  (airData.CO_ppm ? `CO: ${airData.CO_ppm.toFixed(2)} ppm, ` : '') +
                                  (airData.O3_ppb ? `O3: ${airData.O3_ppb.toFixed(1)} ppb` : '');
                } else if (this.prediction.pm25) {
                    pm25Display = `PM2.5: ${this.prediction.pm25} μg/m³`;
                }

                const aqiClass = this.getAQIClass(aqi);
                const aqiLevel = this.getAQILevel(aqi);

                const sources = this.prediction.sources_used || [];
                const sourcesHTML = sources.length > 0 
                    ? `<div class="data-sources">📡 Источники: ${sources.join(', ')}</div>`
                    : '';

                document.getElementById('currentAqi').innerHTML = `
                    <div class="aqi-card ${aqiClass}">
                        <h2>AQI: ${aqi}</h2>
                        <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                            ${pm25Display}
                        </p>
                        <p style="opacity: 0.9; margin-top: 0.5rem;">
                            ${aqiLevel.text}
                        </p>
                        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;">
                            Индекс загрязнения: ${pollutionIndex.toFixed(1)}
                        </p>
                        ${sourcesHTML}
                    </div>
                `;

                const adviceHTML = (this.prediction.advice || [])
                    .map(adv => `<li>${adv}</li>`)
                    .join('');
                document.getElementById('adviceList').innerHTML = adviceHTML || 
                    '<li>Нет специальных рекомендаций</li>';
            },

            displayWeights() {
                if (!this.prediction || !this.prediction.dynamic_weights) {
                    document.getElementById('weightsDisplay').innerHTML = '<em>Нет данных по весам.</em>';
                    return;
                }

                const weights = this.prediction.dynamic_weights;
                const weightsArray = Object.entries(weights)
                    .sort((a, b) => b[1] - a[1]);

                const weightsHTML = weightsArray.map(([factor, weight]) => {
                    const percentage = (weight * 100).toFixed(1);
                    const label = this.getFactorLabel(factor);
                    
                    return `
                        <div class="weight-bar">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem; font-size: 0.9rem;">
                                <span>${label}</span>
                                <span><strong>${percentage}%</strong></span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                                <div class="weight-bar-fill" style="width: ${percentage}%">
                                    ${percentage}%
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                document.getElementById('weightsDisplay').innerHTML = weightsHTML + 
                    '<p style="margin-top: 1rem; font-size: 0.85rem; color: #666;">⚙️ Веса автоматически адаптируются к текущим условиям</p>';
            },

            getFactorLabel(factor) {
                const labels = {
                    'AvgTemperature_C': '🌡️ Температура',
                    'AvgWindSpeed_m_s': '💨 Скорость ветра',
                    'AvgPrecipitation_mm': '🌧️ Осадки',
                    'CO_ppm': '🏭 CO (угарный газ)',
                    'NO2_ppb': '🚗 NO2 (диоксид азота)',
                    'O3_ppb': '☀️ O3 (озон)',
                    'TrafficIndex': '🚦 Интенсивность трафика'
                };
                return labels[factor] || factor;
            },

            getAQIClass(aqi) {
                if (aqi <= 50) return 'aqi-good';
                if (aqi <= 100) return 'aqi-moderate';
                if (aqi <= 150) return 'aqi-unhealthy-sensitive';
                if (aqi <= 200) return 'aqi-unhealthy';
                return 'aqi-very-unhealthy';
            },

            getAQILevel(aqi) {
                if (aqi <= 50) return { text: 'Хорошо', color: '#00e400' };
                if (aqi <= 100) return { text: 'Умеренно', color: '#ffff00' };
                if (aqi <= 150) return { text: 'Нездорово для чувствительных', color: '#ff7e00' };
                if (aqi <= 200) return { text: 'Нездорово', color: '#ff0000' };
                return { text: 'Очень нездорово', color: '#8f3f97' };
            },

            showError(message) {
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = '❌ ' + message;
                errorDiv.classList.remove('hidden');
                document.getElementById('loading').classList.add('hidden');
            },

            hideError() {
                document.getElementById('error').classList.add('hidden');
            },

            // ----------------- Forecast functions -----------------
            async getForecast() {
                if (!this.location) {
                    this.showError('Местоположение не определено');
                    return;
                }

                try {
                    document.getElementById('forecastInfo').textContent = '⏳ Загружаем прогноз...';
                    const res = await fetch(`${this.backendUrl}/forecast`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lat: this.location.lat, lon: this.location.lon })
                    });

                    if (!res.ok) {
                        const err = await res.json().catch(()=>({detail:'API error'}));
                        throw new Error(err.detail || 'Ошибка API');
                    }

                    const data = await res.json();
                    this.forecastData = data;
                    this.renderForecastChart(data);
                    this.renderWeatherTable(data.forecast || []);
                    document.getElementById('forecastInfo').textContent = `Последнее обновление: ${new Date(data.timestamp).toLocaleString()}`;
                    this.hideError();
                } catch (e) {
                    console.error('Forecast error:', e);
                    this.showError('Ошибка получения прогноза на неделю: ' + e.message);
                    document.getElementById('forecastInfo').textContent = '';
                }
            },

            renderForecastChart(data) {
                const forecast = data.forecast || [];
                // Если API возвращает массив почасовых записей с полем time и aqi/predicted_aqi
                const labels = forecast.map(f => {
                    try { return new Date(f.time).toLocaleString('ru-RU', { day: '2-digit', hour: '2-digit' }); }
                    catch { return f.time || ''; }
                });

                const aqiValues = forecast.map(f => (f.aqi ?? f.predicted_aqi ?? f.predicted_aqi_hourly ?? null) || null);

                // если все null — пробуем взять aggregated predicted_aqi если есть
                const anyNonNull = aqiValues.some(v => v !== null);
                const dataset = anyNonNull ? aqiValues : forecast.map((_,i)=>null);

                // Destroy previous chart
                if (this.forecastChart) this.forecastChart.destroy();

                const ctx = document.getElementById('forecastChart').getContext('2d');
                this.forecastChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [
                            {
                                label: 'AQI (прогноз)',
                                data: dataset,
                                tension: 0.25,
                                borderWidth: 2,
                                borderColor: '#667eea',
                                pointRadius: 3,
                                spanGaps: true,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'AQI' }
                            },
                            x: {
                                ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 20 }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => {
                                        const idx = ctx.dataIndex;
                                        const f = (this.forecastData && this.forecastData.forecast && this.forecastData.forecast[idx]) || null;
                                        if (!f) return `AQI: ${ctx.formattedValue}`;
                                        const parts = [`AQI: ${ctx.formattedValue}`];
                                        if (f.temperature) parts.push(`T: ${f.temperature}°C`);
                                        if (f.wind_speed) parts.push(`Wind: ${f.wind_speed} m/s`);
                                        if (f.precipitation) parts.push(`Precip: ${f.precipitation} mm`);
                                        return parts.join(' | ');
                                    }
                                }
                            }
                        }
                    }
                });
            },

            renderWeatherTable(hourlyForecast) {
                // hourlyForecast is an array of hourly points (could be 168 entries)
                if (!hourlyForecast || hourlyForecast.length === 0) {
                    document.getElementById('weatherTable').innerHTML = '<em>Нет погодных данных</em>';
                    return;
                }

                // Build a compact table: дата/час | temp | wind | precip
                let html = '<table style="width:100%; border-collapse: collapse;">';
                html += '<thead><tr style="background:#f5f5f5;"><th style="padding:8px; text-align:left;">Время</th><th style="padding:8px; text-align:right;">T °C</th><th style="padding:8px; text-align:right;">Ветер м/с</th><th style="padding:8px; text-align:right;">Осадки мм</th></tr></thead><tbody>';

                // show every 3rd hour to reduce height or first 56 hours / customizable
                const step = Math.max(1, Math.floor(hourlyForecast.length / 56));
                for (let i = 0; i < hourlyForecast.length; i += step) {
                    const h = hourlyForecast[i];
                    const time = h.time ? new Date(h.time).toLocaleString('ru-RU', { day:'2-digit', hour:'2-digit' }) : h.time || '';
                    html += `<tr><td style="padding:8px;">${time}</td><td style="padding:8px; text-align:right;">${(h.temperature ?? '—')}</td><td style="padding:8px; text-align:right;">${(h.wind_speed ?? '—')}</td><td style="padding:8px; text-align:right;">${(h.precipitation ?? '—')}</td></tr>`;
                }

                html += '</tbody></table>';
                document.getElementById('weatherTable').innerHTML = html;
            },

            downloadForecastCSV() {
                if (!this.forecastData || !this.forecastData.forecast) {
                    this.showError('Нет данных для скачивания');
                    return;
                }
                const rows = this.forecastData.forecast;
                const header = ['time', 'aqi', 'temperature', 'wind_speed', 'precipitation', 'humidity'];
                const csv = [header.join(',')].concat(
                    rows.map(r => {
                        const time = `"${(r.time||'')}"`;
                        const aqi = (r.aqi ?? r.predicted_aqi ?? '');
                        const t = (r.temperature ?? '');
                        const w = (r.wind_speed ?? '');
                        const p = (r.precipitation ?? '');
                        const h = (r.humidity ?? '');
                        return [time, aqi, t, w, p, h].join(',');
                    })
                ).join('\n');

                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `forecast_${new Date().toISOString().slice(0,10)}.csv`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            }
        };

        window.addEventListener('DOMContentLoaded', () => app.init());
    </script>
</body>
</html>

</template>

<script setup>
import AQIGraph from '../components/AQIGraph.vue'
import { ref } from 'vue'
const user = ref({ name: 'Пользователь' })
</script>
