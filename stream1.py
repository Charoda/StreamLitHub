import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.metrics import r2_score
import requests
from io import StringIO
import sys
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ---------- функции обработки ----------
def extract_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r'(\d+\.?\d*)', str(value))
    return float(match.group(1)) if match else np.nan

class CleanAndImputeColumns:
    def __init__(self, columns):
        self.columns = columns
        self.medians_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            cleaned = X[col].apply(extract_number)
            self.medians_[col] = cleaned.median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            cleaned = X_copy[col].apply(extract_number)
            cleaned = cleaned.fillna(self.medians_[col])
            X_copy[col] = cleaned
        return X_copy

def drop_columns(X):
    return X.drop(['name', 'torque'], axis=1, errors='ignore')

def clean_and_impute(df, columns, medians):
    df = df.copy()
    for col in columns:
        cleaned = df[col].apply(extract_number)
        cleaned = cleaned.fillna(medians[col])
        df[col] = cleaned
    return df

def preprocess_data(df, num_cols, cat_cols, scaler, encoder, medians):
    df = df.copy()
    df.drop(['name', 'torque'], axis=1, errors='ignore', inplace=True)

    cols_to_clean = ['mileage', 'engine', 'max_power', 'seats']
    df = clean_and_impute(df, cols_to_clean, medians)

    X_num = df[num_cols].values
    X_num_scaled = scaler.transform(X_num)
    df_num_scaled = pd.DataFrame(X_num_scaled, columns=num_cols, index=df.index)

    X_cat = df[cat_cols].values
    X_cat_encoded = encoder.transform(X_cat)
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    df_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)

    df_final = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    return df_final

def plot_coefficients_pie(coef, feature_names, title="Абсолютные значения коэффициентов модели"):
    if len(coef) != len(feature_names):
        return None
    coef_df = pd.DataFrame({
        'Признак': feature_names,
        'Коэффициент': coef
    })
    coef_df['abs_coef'] = np.abs(coef_df['Коэффициент'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    top_n = 15
    if len(coef_df) > top_n:
        top = coef_df.head(top_n)
        other = coef_df.iloc[top_n:]
        other_sum = other['abs_coef'].sum()
        other_row = pd.DataFrame({
            'Признак': ['Остальные признаки'],
            'Коэффициент': [other['Коэффициент'].sum()],
            'abs_coef': [other_sum]
        })
        coef_df_plot = pd.concat([top, other_row], ignore_index=True)
    else:
        coef_df_plot = coef_df

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['green' if c > 0 else 'red' for c in coef_df_plot['Коэффициент']]
    for i, name in enumerate(coef_df_plot['Признак']):
        if name == 'Остальные признаки':
            colors[i] = 'gray'

    wedges, texts, autotexts = ax.pie(
        coef_df_plot['abs_coef'],
        labels=coef_df_plot['Признак'],
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 8}
    )
    ax.set_title(title, fontsize=12)
    legend_elements = [
        Patch(facecolor='green', label='Положительный вклад (увеличивает цену)'),
        Patch(facecolor='red', label='Отрицательный вклад (уменьшает цену)'),
        Patch(facecolor='gray', label='Остальные признаки (суммарно)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
    return fig

# ---------- Инициализация состояния сессии ----------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'inference_objects' not in st.session_state:
    st.session_state['inference_objects'] = None
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'show_weights' not in st.session_state:
    st.session_state['show_weights'] = False
if 'show_weights_main' not in st.session_state:
    st.session_state['show_weights_main'] = False
if 'analysis_df' not in st.session_state:
    st.session_state['analysis_df'] = None
if 'show_corr' not in st.session_state:
    st.session_state['show_corr'] = False
if 'show_hist' not in st.session_state:
    st.session_state['show_hist'] = False
if 'show_cat' not in st.session_state:
    st.session_state['show_cat'] = False
if 'show_describe' not in st.session_state:
    st.session_state['show_describe'] = False
if 'analysis_loaded' not in st.session_state:
    st.session_state['analysis_loaded'] = False

st.set_page_config(page_title="Предсказание цены автомобиля", layout="wide")
st.title("Предсказание цены автомобиля")
st.markdown("Загрузите данные для анализа, затем загрузите модель для предсказаний.")

# ---------- Раздел анализа данных (до загрузки модели) ----------
st.subheader("Анализ данных (загрузите данные для визуализации)")
analysis_col1, analysis_col2 = st.columns(2)
with analysis_col1:
    analysis_file = st.file_uploader("Загрузите CSV-файл для анализа", type=["csv"], key="analysis_uploader")
    if analysis_file is not None:
        try:
            df_temp = pd.read_csv(analysis_file)
            st.session_state['analysis_df'] = df_temp
            st.session_state['analysis_loaded'] = True
            st.success("Данные для анализа загружены из файла!")
            print("Анализ: данные загружены из файла, строк:", len(df_temp))
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            print("Ошибка чтения файла для анализа:", e)

with analysis_col2:
    st.caption("Или укажите ссылку:")
    analysis_url = st.text_input("URL CSV-файла для анализа", value="https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv", key="analysis_url")
    if st.button("Загрузить данные по ссылке (анализ)"):
        try:
            response = requests.get(analysis_url)
            response.raise_for_status()
            df_temp = pd.read_csv(StringIO(response.text))
            st.session_state['analysis_df'] = df_temp
            st.session_state['analysis_loaded'] = True
            st.success("Данные для анализа загружены по ссылке!")
            print("Анализ: данные загружены по ссылке, строк:", len(df_temp))
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            print("Ошибка загрузки данных для анализа:", e)

analysis_df = st.session_state['analysis_df']
if analysis_df is not None and st.session_state.get('analysis_loaded', False):
    st.subheader("Загруженные данные (анализ)")
    st.dataframe(analysis_df.head(10))
    st.caption(f"Всего строк: {len(analysis_df)}")

    # Четыре кнопки для описания и графиков
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("Показать/скрыть описание данных"):
            st.session_state['show_describe'] = not st.session_state.get('show_describe', False)
            print(f"Описание данных: {st.session_state['show_describe']}")
    with col_btn2:
        if st.button("Показать/скрыть корреляционную матрицу"):
            st.session_state['show_corr'] = not st.session_state.get('show_corr', False)
            print(f"Корреляционная матрица: {st.session_state['show_corr']}")
    with col_btn3:
        if st.button("Показать/скрыть гистограммы"):
            st.session_state['show_hist'] = not st.session_state.get('show_hist', False)
            print(f"Гистограммы: {st.session_state['show_hist']}")
    with col_btn4:
        if st.button("Показать/скрыть категориальные графики"):
            st.session_state['show_cat'] = not st.session_state.get('show_cat', False)
            print(f"Категориальные графики: {st.session_state['show_cat']}")

    # Обработка данных для анализа (очистка и заполнение) – кэшируется
    @st.cache_data
    def prepare_analysis_data(df):
        df_clean = df.drop(['name', 'torque'], axis=1, errors='ignore')
        cols_to_clean = ['mileage', 'engine', 'max_power', 'seats']
        medians_analysis = {}
        for col in cols_to_clean:
            if col in df_clean.columns:
                cleaned = df_clean[col].apply(extract_number)
                medians_analysis[col] = cleaned.median()
        df_clean = clean_and_impute(df_clean, cols_to_clean, medians_analysis)
        return df_clean

    try:
        df_clean = prepare_analysis_data(analysis_df)
        num_cols_analysis = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols_analysis = df_clean.select_dtypes(include=['object']).columns.tolist()
        if 'selling_price' in num_cols_analysis:
            num_cols_analysis.remove('selling_price')
        if 'selling_price' in cat_cols_analysis:
            cat_cols_analysis.remove('selling_price')

        # ---- Описание данных ----
        if st.session_state.get('show_describe', False):
            st.subheader("Общая информация о данных")
            # Общая информация
            info_lines = []
            info_lines.append(f"Количество строк: {len(df_clean)}")
            info_lines.append(f"Количество колонок: {len(df_clean.columns)}")
            info_lines.append(f"Числовые признаки: {len(num_cols_analysis)}")
            info_lines.append(f"Категориальные признаки: {len(cat_cols_analysis)}")
            st.write("\n".join(info_lines))

            st.subheader("Описательная статистика для числовых признаков")
            st.dataframe(df_clean[num_cols_analysis].describe())

            if cat_cols_analysis:
                st.subheader("Частотные таблицы для категориальных признаков (топ-5 значений)")
                for col in cat_cols_analysis:
                    st.write(f"**{col}**")
                    freq = df_clean[col].value_counts().head(5)
                    st.dataframe(freq.reset_index().rename(columns={'index': col, col: 'Количество'}))

        # 1. Корреляционная матрица
        if st.session_state.get('show_corr', False):
            if len(num_cols_analysis) >= 2:
                print("Построение корреляционной матрицы...")
                st.subheader("Корреляционная матрица (числовые признаки)")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                corr = df_clean[num_cols_analysis].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, linewidths=0.5)
                ax_corr.set_title("Корреляция числовых признаков")
                st.pyplot(fig_corr)
            else:
                st.info("Недостаточно числовых признаков для корреляционной матрицы.")

        # 2. Гистограммы
        if st.session_state.get('show_hist', False):
            if num_cols_analysis:
                print("Построение гистограмм...")
                st.subheader("Гистограммы числовых признаков")
                n_cols = min(3, len(num_cols_analysis))
                n_rows = (len(num_cols_analysis) + n_cols - 1) // n_cols
                fig_hist, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                for i, col in enumerate(num_cols_analysis):
                    if i < len(axes):
                        df_clean[col].hist(ax=axes[i], bins=30, edgecolor='black', alpha=0.7)
                        axes[i].set_title(col)
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("Частота")
                for j in range(len(num_cols_analysis), len(axes)):
                    axes[j].set_visible(False)
                fig_hist.tight_layout()
                st.pyplot(fig_hist)

        # 3. Категориальные графики
        if st.session_state.get('show_cat', False):
            if cat_cols_analysis:
                print("Построение категориальных графиков...")
                st.subheader("Распределение категориальных признаков")
                n_cat = len(cat_cols_analysis)
                n_cols_cat = min(2, n_cat)
                n_rows_cat = (n_cat + n_cols_cat - 1) // n_cols_cat
                fig_cat, axes_cat = plt.subplots(n_rows_cat, n_cols_cat, figsize=(12, 5 * n_rows_cat))
                if n_rows_cat == 1 and n_cols_cat == 1:
                    axes_cat = [axes_cat]
                else:
                    axes_cat = axes_cat.flatten()
                for i, col in enumerate(cat_cols_analysis):
                    if i < len(axes_cat):
                        counts = df_clean[col].value_counts()
                        if len(counts) > 10:
                            counts = counts.head(10)
                            counts['Остальные'] = len(df_clean[col]) - counts.sum()
                        counts.plot(kind='bar', ax=axes_cat[i], color='skyblue', edgecolor='black')
                        axes_cat[i].set_title(col)
                        axes_cat[i].set_xlabel(col)
                        axes_cat[i].set_ylabel("Количество")
                        axes_cat[i].tick_params(axis='x', rotation=45)
                for j in range(len(cat_cols_analysis), len(axes_cat)):
                    axes_cat[j].set_visible(False)
                fig_cat.tight_layout()
                st.pyplot(fig_cat)

    except Exception as e:
        st.error(f"Ошибка при построении графиков или описания: {e}")
        print("Ошибка:", sys.exc_info())
        traceback.print_exc()

st.markdown("---")

# ---------- Загрузка модели и предсказания (осталось без изменений) ----------
st.subheader("Загрузка модели inference_objects.pkl")
inference_file = st.file_uploader("Выберите .pkl файл с объектами (модель, scaler, encoder, medians, cat_cols, num_cols)", type=["pkl"])

if inference_file is not None:
    try:
        print("\n=== Загрузка inference_objects.pkl ===")
        with st.spinner("Загрузка объектов..."):
            objects = pickle.load(inference_file)
        print("Тип загруженного объекта:", type(objects))
        if isinstance(objects, dict):
            print("Ключи словаря:", list(objects.keys()))
        else:
            print("Объект не является словарём!")

        required_keys = ['model', 'scaler', 'encoder', 'medians', 'cat_cols', 'num_cols']
        if isinstance(objects, dict):
            missing_keys = [key for key in required_keys if key not in objects]
            if missing_keys:
                st.error(f"В загруженном файле отсутствуют ключи: {missing_keys}")
                print("Отсутствуют ключи:", missing_keys)
                st.stop()
            else:
                st.success("Все необходимые ключи найдены!")
                st.session_state['inference_objects'] = objects
                print("Объекты успешно извлечены.")
                st.success("Объекты загружены и проверены.")
        else:
            st.error("Загруженный объект не является словарём.")
            print("Ошибка: объект не словарь.")
            st.stop()
    except Exception as e:
        st.error(f"Ошибка загрузки или проверки файла: {e}")
        print("Детали ошибки:", sys.exc_info())
        traceback.print_exc()
        st.stop()
else:
    if st.session_state['inference_objects'] is not None:
        objects = st.session_state['inference_objects']
        st.info("Объекты уже загружены из сессии.")
        print("Используем объекты из session_state")
    else:
        st.warning("Пожалуйста, загрузите inference_objects.pkl, чтобы продолжить.")
        st.stop()

objects = st.session_state['inference_objects']
model = objects['model']
scaler = objects['scaler']
encoder = objects['encoder']
medians = objects['medians']
cat_cols = objects['cat_cols']
num_cols = objects['num_cols']

# ---- Блок анализа весов модели ----
st.subheader("Анализ весов модели")
if st.button("Показать значимость весов (без данных)"):
    st.session_state['show_weights_main'] = not st.session_state.get('show_weights_main', False)
    print(f"Показ весов: {st.session_state['show_weights_main']}")

if st.session_state.get('show_weights_main', False):
    coef = model.coef_
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_feature_names)
    fig = plot_coefficients_pie(coef, all_feature_names, "Вклад признаков (веса модели)")
    if fig is not None:
        with st.container():
            st.pyplot(fig, use_container_width=False)
    else:
        st.warning("Не удалось построить диаграмму из-за несовпадения размерностей.")

st.markdown("---")

# ---- Загрузка данных для предсказания (тестовые) ----
st.subheader("Загрузите данные для предсказания")
uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"], key="file_uploader")

st.markdown("---")
st.caption("Или укажите прямую ссылку на CSV-файл:")
default_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
url = st.text_input("Введите URL CSV-файла", value=default_url, placeholder="https://example.com/data.csv")

col1, col2 = st.columns(2)
with col1:
    if st.button("Загрузить по ссылке"):
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            st.session_state['df'] = df
            st.success("Данные загружены по ссылке!")
            print("Данные загружены по ссылке, строк:", len(df))
        except Exception as e:
            st.error(f"Ошибка загрузки данных по ссылке: {e}")
            print("Ошибка загрузки по ссылке:", e)
with col2:
    if st.button("Загрузить тестовые данные (cars_test.csv)"):
        try:
            response = requests.get(default_url)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            st.session_state['df'] = df
            st.success("Тестовые данные загружены!")
            print("Тестовые данные загружены, строк:", len(df))
        except Exception as e:
            st.error(f"Ошибка загрузки тестовых данных: {e}")
            print("Ошибка загрузки тестовых данных:", e)

df = st.session_state['df']

if df is not None:
    st.subheader("Загруженные данные (для предсказания)")
    st.dataframe(df.head(10))
    st.caption(f"Всего строк: {len(df)}")

    has_target = 'selling_price' in df.columns

    if st.button("Предсказать цены"):
        print("\n=== Кнопка 'Предсказать цены' нажата ===")

        if df is None:
            st.error("Данные не загружены. Пожалуйста, загрузите файл или укажите ссылку.")
            print("Ошибка: df is None")
            st.stop()

        if model is None or scaler is None or encoder is None or medians is None:
            st.error("Не все объекты для инференса загружены. Пожалуйста, загрузите inference_objects.pkl.")
            print("Ошибка: один из объектов None")
            st.stop()

        try:
            print("Начинаем предобработку...")
            with st.spinner("Применение предобработки..."):
                X_prepared = preprocess_data(df, num_cols, cat_cols, scaler, encoder, medians)
            print("Предобработка завершена, форма:", X_prepared.shape)
            st.success("Данные обработаны")
            st.write("Пример обработанных данных (первые 5 строк):")
            st.dataframe(X_prepared.head(5))

            with st.spinner("Выполнение предсказания..."):
                predictions = model.predict(X_prepared)
            print("Предсказание выполнено")

            df_result = df.copy()
            df_result['predicted_price'] = predictions

            r2_value = None
            if has_target:
                r2_value = r2_score(df['selling_price'], predictions)

            st.session_state['prediction_result'] = {
                'df_result': df_result,
                'predictions': predictions,
                'r2': r2_value,
                'has_target': has_target,
                'coef': model.coef_,
                'all_feature_names': list(num_cols) + list(encoder.get_feature_names_out(cat_cols)),
                'cat_feature_names': encoder.get_feature_names_out(cat_cols),
                'num_cols': num_cols,
                'cat_cols': cat_cols
            }

            st.subheader("Результаты предсказания")
            st.dataframe(df_result)
            if has_target and r2_value is not None:
                st.metric("Коэффициент детерминации (R²) на загруженных данных", f"{r2_value:.4f}")

            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

            st.markdown("---")
            if st.button("Показать значимость весов (после предсказания)"):
                st.session_state['show_weights'] = not st.session_state.get('show_weights', False)
                print(f"Показ весов (после предсказания): {st.session_state['show_weights']}")

        except Exception as e:
            st.error(f"Ошибка во время предсказания: {e}")
            print("Исключение в блоке предсказания:", sys.exc_info())
            traceback.print_exc()

    if st.session_state.get('show_weights', False) and st.session_state['prediction_result'] is not None:
        result = st.session_state['prediction_result']
        coef = result['coef']
        all_feature_names = result['all_feature_names']
        fig = plot_coefficients_pie(coef, all_feature_names)
        if fig is not None:
            with st.container():
                st.pyplot(fig, use_container_width=False)
        else:
            st.warning("Не удалось построить диаграмму.")

else:
    st.info("Ожидание загрузки данных для предсказания (файл, ссылка или кнопка 'Загрузить тестовые данные').")

with st.expander("Требуемый формат файла"):
    st.markdown("""
    CSV-файл должен содержать следующие колонки (названия **точно** такие же):
    - `year` (год выпуска, целое число)
    - `km_driven` (пробег, целое число)
    - `mileage` (расход топлива, может содержать суффикс, например "23.4 kmpl")
    - `engine` (объём двигателя, может содержать "CC")
    - `max_power` (мощность, может содержать "bhp")
    - `fuel` (тип топлива: "Diesel", "Petrol", "LPG" и т.д.)
    - `seller_type` (тип продавца: "Individual", "Trustmark Dealer")
    - `transmission` (коробка передач: "Manual", "Automatic")
    - `owner` (количество владельцев: "First Owner", "Second Owner" и т.д.)
    - `seats` (количество мест, число, например 5.0)

    Колонка `selling_price` (фактическая цена) **необязательна** – если она есть, будет вычислен R².
    Колонки `name` и `torque` могут присутствовать, но будут проигнорированы.
    """)
    st.code("""year,km_driven,mileage,engine,max_power,fuel,seller_type,transmission,owner,seats,selling_price
2014,145500,23.4 kmpl,1248 CC,74 bhp,Diesel,Individual,Manual,First Owner,5.0,450000
2017,45000,20.14 kmpl,1197 CC,81.86 bhp,Petrol,Individual,Manual,First Owner,5.0,440000
""", language='csv')