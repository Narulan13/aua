<template>
  <div class="min-h-screen bg-gradient-to-br from-indigo-100 to-blue-200 flex items-center justify-center">
    <div class="bg-white/80 backdrop-blur-md shadow-2xl rounded-3xl p-10 w-full max-w-md">
      <h2 class="text-3xl font-bold text-center text-indigo-600 mb-6">Регистрация</h2>
      
      <form @submit.prevent="registerUser" class="space-y-4">
        <input v-model="form.name" type="text" placeholder="Имя" class="input" />
        <input v-model="form.year" type="number" placeholder="Год рождения" class="input" />
        <input v-model="form.family_information" type="text" placeholder="Семейная информация" class="input" />
        <input v-model="form.health" type="text" placeholder="Здоровье" class="input" />
        <input v-model="form.email" type="email" placeholder="Email" class="input" />
        <input v-model="form.password" type="password" placeholder="Пароль" class="input" />
        
        <button type="submit" class="btn-primary w-full">Создать аккаунт</button>
      </form>

      <p class="text-center text-gray-500 mt-4">
        Уже есть аккаунт? <router-link to="/login" class="text-indigo-600 font-semibold">Войти</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import axios from '../api.js'
import { reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const form = reactive({
  name: '', year: '', family_information: '', health: '', email: '', password: ''
})

const registerUser = async () => {
  await axios.post('/auth/register', form)
  alert('✅ Успешная регистрация!')
  router.push('/login')
}
</script>

<style scoped>
.input {
  @apply w-full px-4 py-3 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:outline-none;
}
.btn-primary {
  @apply bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl transition-all duration-200;
}
</style>
