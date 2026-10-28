# -*- coding: utf-8 -*-
"""
Despliegue - Predicción de la frecuencia cardíaca máxima (MaxHR)
- Cargamos el modelo (Pipeline: StandardScaler + Regresión Lineal)
- Capturamos los datos del paciente
- Preparamos los datos: dummies y columnas en el orden del entrenamiento
- Aplicamos el modelo para la predicción
"""

# Librerías
import pickle
import pandas as pd
import streamlit as st

# Cargamos el modelo
# El .pkl contiene [modelo, variables]. La normalización va dentro del Pipeline,
# por eso no se carga un scaler aparte ni se llama transform manualmente.
filename = 'modelo-lr-reg.pkl'
with open(filename, 'rb') as f:
    modelo, variables = pickle.load(f)

# Interfaz gráfica
st.title('Predicción de la frecuencia cardíaca máxima (MaxHR)')
st.write('Estimación de la frecuencia cardíaca máxima alcanzada en la prueba de esfuerzo '
         'a partir de datos clínicos del paciente.')

col1, col2 = st.columns(2)

with col1:
    Age = st.slider('Edad (años)', min_value=28, max_value=77, value=54, step=1)
    Sex = st.selectbox('Sexo', ['M', 'F'], format_func=lambda x: 'Masculino' if x == 'M' else 'Femenino')
    RestingBP = st.slider('Presión arterial en reposo (mmHg)', min_value=80, max_value=200, value=130, step=1)
    FastingBS = st.selectbox('Glucosa en ayunas > 120 mg/dl', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
    Oldpeak = st.slider('Oldpeak (depresión del ST, mm)', min_value=-2.6, max_value=6.2, value=0.6, step=0.1)

with col2:
    ChestPainType = st.selectbox('Tipo de dolor de pecho', ['ASY', 'ATA', 'NAP', 'TA'],
                                 format_func=lambda x: {'ASY': 'ASY - Asintomático',
                                                        'ATA': 'ATA - Angina atípica',
                                                        'NAP': 'NAP - Dolor no anginoso',
                                                        'TA': 'TA - Angina típica'}[x])
    RestingECG = st.selectbox('ECG en reposo', ['Normal', 'ST', 'LVH'],
                              format_func=lambda x: {'Normal': 'Normal',
                                                     'ST': 'ST - Anomalía onda ST-T',
                                                     'LVH': 'LVH - Hipertrofia ventricular izq.'}[x])
    ExerciseAngina = st.selectbox('Angina inducida por ejercicio', ['N', 'Y'],
                                  format_func=lambda x: 'Sí' if x == 'Y' else 'No')
    ST_Slope = st.selectbox('Pendiente del segmento ST', ['Up', 'Flat', 'Down'],
                            format_func=lambda x: {'Up': 'Up - Ascendente',
                                                   'Flat': 'Flat - Plana',
                                                   'Down': 'Down - Descendente'}[x])

# Dataframe con los mismos nombres de variables del dataset original
datos = [[Age, Sex, ChestPainType, RestingBP, FastingBS, RestingECG, ExerciseAngina, Oldpeak, ST_Slope]]
data = pd.DataFrame(datos, columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'FastingBS',
                                    'RestingECG', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

# Preparación de datos
data_preparada = data.copy()

# En despliegue drop_first=False: con un solo registro, drop_first eliminaría la única categoría presente
data_preparada = pd.get_dummies(data_preparada,
                                columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
                                drop_first=False, dtype=int)

# Se ajustan las columnas al orden del entrenamiento: se agregan las faltantes en 0 y se eliminan
# las categorías de referencia (Sex_F, ChestPainType_ASY, RestingECG_LVH, ExerciseAngina_N, ST_Slope_Down)
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# Predicción (el Pipeline estandariza Age, RestingBP y Oldpeak internamente; en despliegue no se llama fit)
if st.button('Predecir MaxHR', type='primary'):
    Y_pred = modelo.predict(data_preparada)[0]
    fc_teorica = 220 - Age

    st.success(f'MaxHR estimada: **{Y_pred:.0f} lpm**')
    st.write(f'Referencia: la FC máxima teórica por edad (220 − edad) es **{fc_teorica} lpm**. '
             f'La predicción equivale al {Y_pred / fc_teorica * 100:.0f}% de ese valor.')

    with st.expander('Ver datos enviados al modelo'):
        st.dataframe(data_preparada)

# Recordar medida de error del modelo
st.warning('El modelo tiene un error promedio (MAE) de ±16.7 lpm (MAPE ≈ 13%). '
           'Es más preciso en el rango típico (120–160 lpm) y menos confiable en valores extremos. '
           'Uso académico: no reemplaza la evaluación médica.')
