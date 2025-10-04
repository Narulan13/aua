<template>
  <div id="app">
    <header class="header">
      <h1>🌍 AirQualityAI</h1>
      <p class="subtitle">Персональный мониторинг качества воздуха</p>
    </header>

    <main class="container">
      <div v-if="loading" class="loading">
        <div class="spinner"></div>
        <p>Определяем ваше местоположение...</p>
      </div>

      <div v-else-if="error" class="error">
        <p>❌ {{ error }}</p>
        <button @click="requestLocation" class="btn">Попробовать снова</button>
      </div>

      <div v-else class="content">
        <!-- User Profile Section -->
        <div class="profile-section">
          <h3>👤 Ваш профиль</h3>
          <div class="profile-inputs">
            <label>
              Возраст:
              <input v-model.number="profile.age" type="number" min="1" max="120" />
            </label>
            <label>
              <input v-model="profile.asthma" type="checkbox" />
              У меня астма
            </label>
          </div>
          <button @click="getPrediction" class="btn btn-primary">
            Обновить прогноз
          </button>
        </div>

        <!-- Results Section -->
        <div v-if="prediction" class="results">
          <div class="aqi-card" :class="aqiClass">
            <h2>AQI: {{ prediction.aqi }}</h2>
            <p class="pm25">PM2.5: {{ prediction.pm25 }} μg/m³</p>
            <p class="confidence">Точность: {{ (prediction.confidence * 100).toFixed(0) }}%</p>
          </div>

          <div class="advice-section">
            <h3>💡 Рекомендации</h3>
            <ul>
              <li v-for="(adv, idx) in prediction.advice" :key="idx">
                {{ adv }}
              </li>
            </ul>
          </div>

          <div class="location-info">
            <p>📍 Координаты: {{ prediction.location.lat.toFixed(4) }}, {{ prediction.location.lon.toFixed(4) }}</p>
          </div>
        </div>

        <!-- Map Component -->
        <MapView 
          v-if="location" 
          :lat="location.lat" 
          :lon="location.lon"
          :aqi="prediction ? prediction.aqi : null"
        />
      </div>
    </main>

    <footer class="footer">
      <p>MVP v1.0 | Данные обновляются каждые 15 минут</p>
    </footer>
  </div>
</template>

<script>
import MapView from './components/MapView.vue'

export default {
  name: 'App',
  components: {
    MapView
  },
  data() {
    return {
      location: null,
      prediction: null,
      loading: true,
      error: null,
      profile: {
        age: 30,
        asthma: false
      }
    }
  },
  computed: {
    aqiClass() {
      if (!this.prediction) return ''
      const aqi = this.prediction.aqi
      if (aqi <= 50) return 'aqi-good'
      if (aqi <= 100) return 'aqi-moderate'
      if (aqi <= 150) return 'aqi-unhealthy-sensitive'
      if (aqi <= 200) return 'aqi-unhealthy'
      return 'aqi-very-unhealthy'
    }
  },
  mounted() {
    this.requestLocation()
  },
  methods: {
    requestLocation() {
      this.loading = true
      this.error = null

      if (!navigator.geolocation) {
        this.error = 'Геолокация не поддерживается вашим браузером'
        this.loading = false
        return
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          this.location = {
            lat: position.coords.latitude,
            lon: position.coords.longitude
          }
          this.loading = false
          this.getPrediction()
        },
        (err) => {
          this.error = 'Не удалось определить местоположение. Пожалуйста, разрешите доступ к геолокации.'
          this.loading = false
          console.error(err)
        }
      )
    },
    async getPrediction() {
      if (!this.location) return

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            lat: this.location.lat,
            lon: this.location.lon,
            timestamp: new Date().toISOString(),
            profile: this.profile
          })
        })

        if (!response.ok) {
          throw new Error('Ошибка получения прогноза')
        }

        this.prediction = await response.json()
      } catch (err) {
        this.error = 'Не удалось получить прогноз. Проверьте, запущен ли backend на порту 8000.'
        console.error(err)
      }
    }
  }
}
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #333;
}

.header {
  background: rgba(255, 255, 255, 0.95);
  padding: 2rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #666;
  font-size: 1.1rem;
}

.container {
  max-width: 1200px;
  margin: 2rem auto;
  padding: 0 1rem;
}

.loading, .error {
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
  gap: 1.5rem;
}

.profile-section, .results {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
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
}

.profile-inputs input[type="number"] {
  width: 80px;
  padding: 0.5rem;
  border: 2px solid #ddd;
  border-radius: 6px;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s;
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
}

.aqi-good { background: #00e400; color: white; }
.aqi-moderate { background: #ffff00; color: #333; }
.aqi-unhealthy-sensitive { background: #ff7e00; color: white; }
.aqi-unhealthy { background: #ff0000; color: white; }
.aqi-very-unhealthy { background: #8f3f97; color: white; }

.aqi-card h2 {
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

.pm25 {
  font-size: 1.3rem;
  margin-bottom: 0.5rem;
}

.confidence {
  opacity: 0.8;
}

.advice-section {
  margin-top: 1rem;
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
}

.location-info {
  margin-top: 1rem;
  padding: 1rem;
  background: #f9f9f9;
  border-radius: 8px;
  text-align: center;
}

.footer {
  text-align: center;
  padding: 2rem;
  color: white;
  opacity: 0.8;
}
</style>