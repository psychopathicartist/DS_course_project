import streamlit as st
import reveal_slides as rs


def presentation_page():
    st.title('Презентация проекта')

    # Содержание презентации в формате Markdown
    presentation_markdown = """
        ## Прогнозирование отказов оборудования
        
        --
        
        ### Введение
        
        #### Задача проекта:
        Разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования или нет.
        
        #### Цель проекта:
        Предсказать отказ оборудования на основе параметров работы
        
        --
        
        ### Введение
        
        #### Датасет:
        "AI4I 2020 Predictive Maintenance Dataset" содержит синтетические данные, моделирующие задачу предиктивного 
        обслуживания оборудования.
        
        - 10 000 записей
        - 14 признаков
        - Бинарная классификация
      
        --
        
        ### Этапы работы
        
        1. Загрузка данных
        2. Предобработка данных
        3. Выбор модели
        4. Обучение модели
        5. Оценка модели
        6. Визуализация результатов
        
        --
        
        ### Streamlit-приложение
        
        **Основная страница:**
        - Анализ данных
        - Предсказания
        - Визуализация метрик
        
        **Страница презентации:**
        - Описание проекта
        - Демонстрация результатов
        
        --
        
        ### Заключение
        
        **Итоги:**
        - Реализована модель предсказания отказов
        - Создано интерактивное приложение
        
        **Возможные улучшения:**
        - Добавление новых моделей
        - Улучшение интерфейса
        """

    # Настройки презентации
    with st.sidebar:
        st.header('Настройки презентации')
        theme = st.selectbox('Тема', ['black', 'white', 'league', 'beige',
                                      'sky', 'night', 'serif', 'simple', 'solarized'])
        height = st.number_input('Высота слайдов', value=500)
        transition = st.selectbox("Переход", ['slide', 'convex', 'concave',
                                              'zoom', 'none'])
        plugins = st.multiselect('Плагины', ['highlight', 'katex', 'mathjax2',
                                             'mathjax3', 'notes', 'search', 'zoom'], [])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={'transition': transition, 'plugins': plugins, 'slideNumber': True, },
        markdown_props={'data-separator-vertical': '^--$'},
        )
