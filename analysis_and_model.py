import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def analysis_and_model_page():
    st.title('Анализ данных и модель')

    if 'model' not in st.session_state:
        st.session_state.model = None

    # Загрузка данных
    uploaded_file = st.file_uploader('Загрузите датасет (CSV)', type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv('data/predictive_maintenance.csv')

    # Удаление ненужных столбцов
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

    # Преобразование категориальной переменной Type в числовую
    data['Type'] = LabelEncoder().fit_transform(data['Type'])

    # Замена пропущенных значений на среднее
    data.fillna(data.mean(), inplace=True)

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Выбор модели для обучения
    st.write('Выберите модель для обучения. Рекомендуемая модель - метод лучайного леса.')
    selected_model = st.selectbox('Вид модели', ['Логистическая регрессия',
                                                 'Метод случайного леса',
                                                 'Метод опорных векторов'])

    if st.button('Начать обучение'):
        # Обучение модели
        if selected_model == 'Логистическая регрессия':
            st.session_state.model = LogisticRegression()
            st.session_state. model.fit(X_train, y_train)
        elif selected_model == 'Метод случайного леса':
            st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
            st.session_state.model.fit(X_train, y_train)
        else:
            st.session_state.model = SVC(kernel='linear', random_state=42, probability=True)
            st.session_state.model.fit(X_train, y_train)

        # Оценка модели
        y_pred = st.session_state.model.predict(X_test)
        y_pred_proba = st.session_state.model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Визуализация результатов
        st.header('Результаты обучения модели')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.subheader('Матрица ошибок')
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.subheader('Классификационный отчёт')
        st.text(class_report)
        st.subheader('ROC-AUC')
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{st.session_state.model.__class__.__name__} (AUC = {roc_auc:.2f})")
        st.pyplot(fig)

    if st.session_state.model is not None:
        # Интерфейс для предсказания
        st.header('Предсказание по новым данным')
        with st.form('prediction_form'):
            st.write('Введите значения признаков для предсказания:')
            prod_type = st.selectbox('Type', ['L', 'M', 'H'])
            air_temp = st.number_input('Air temperature [K]')
            process_temp = st.number_input('Process temperature [K]')
            rotational_speed = st.number_input('Rotational speed [rpm]')
            torque = st.number_input('Torque [Nm]')
            tool_wear = st.number_input('Tool wear [min]')
            submit_button = st.form_submit_button('Предсказать')

            if submit_button:
                # Преобразование введенных данных
                input_data = pd.DataFrame({
                    'Type': [prod_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]}
                )
                input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])

                # Предсказание
                prediction = st.session_state.model.predict(input_data)
                prediction_proba = st.session_state.model.predict_proba(input_data)[:, 1]
                if prediction[0] == 1:
                    st.error(f"Прогнозируется отказ оборудования! (вероятность: {prediction_proba[0]:.2%})")
                else:
                    st.success(f"Оборудование в норме. (вероятность отказа: {prediction_proba[0]:.2%})")
