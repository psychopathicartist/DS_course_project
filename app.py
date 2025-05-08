import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка навигации
PAGES = {
    "Презентация": presentation_page,
    "Анализ и модель": analysis_and_model_page
}

# Настройка боковой панели
st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к", list(PAGES.keys()))

# Отображение выбранной страницы
page = PAGES[selection]
page()
