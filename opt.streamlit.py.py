# -*- coding: utf-8 -*-


import numpy as np
import joblib
import streamlit as st

model_filename = "culPred.pkl"

def load_model_and_predict(inputs):
    # Cargamos el modelo
    loaded_model = joblib.load(model_filename)

    # Convertir el diccionario a una matriz 2D usando numpy
    input_array = np.array(list(inputs.values())).astype(float)

    # Ajustar la forma de la matriz a 2D
    input_array = input_array.reshape(1, len(inputs))

    # Usar el modelo para hacer predicciones
    predicted_cult = loaded_model.predict(input_array)[0]

    return predicted_cult

# Colocamos el título
st.title('App Optimizador de cultivos')

# Ingesta de datos para el usuario
N = st.text_input('Niveles de Nitrógeno')
P = st.text_input('Nivel de Fósforo')
K = st.text_input('Nivel de Potasio')
temperature = st.text_input('Nivel de Temperatura')
ph = st.text_input('Nivel de PH')
humidity = st.text_input('Nivel de Humedad')
rainfall = st.text_input('Nivel de Precipitaciones')

# Verificamos si el botón ha sido presionado
if st.button('Predecir cultivo'):
    # Creamos un diccionario con los datos ingresados por el usuario
    user_inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'ph': ph, 'humidity': humidity, 'rainfall': rainfall}

    # Realizamos la predicción
    prediction = load_model_and_predict(user_inputs)

    # Mostramos el resultado
    st.success(f'El cultivo predicho es: {prediction}')

    
    
     
    
