import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import predict_model
from parcial import stacking,max_pooling,padding,stride,convolution
import numpy as np

st.title("API de transformacion pixeles")


uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl',header=None)
        matrix = df.to_numpy()  
        st.write("Matriz de píxeles cargada:")
        st.write(matrix)

        opcion = st.selectbox("Selecciona una transformación", 
                                ["Convolución", "Padding", "Max Pooling", "Stride","Stacking"])

        if opcion == "Convolución":
                st.write("Parámetros para Convolución:")
                
                tipo_de_kernel = st.selectbox("Selecciona el tipo de kernel", 
                                        ["horizontal", "vertical"])
                
                if st.button("Calcular"):
                    
                    resultado = convolution(matrix, tipo_de_kernel)
                    st.write(f"Resultado de la Convolución ({tipo_de_kernel} kernel):")
                    st.write(resultado)

        elif opcion == "Padding":
            st.write("Parámetros para Padding:")
            tamaño_pading = st.number_input("Tamaño del Padding", min_value=0, value=1)
            
            if st.button("Calcular"):
                resultado = padding(matrix, tamaño_pading)
                st.write("Resultado del Padding:")
                st.write(resultado)

        elif opcion == "Max Pooling":
            st.write("Parámetros para Max Pooling:")
            tamañño_max = st.number_input("Tamaño del Pooling", min_value=1, value=2)
            
            if st.button("Calcular"):
                resultado = max_pooling(matrix, tamañño_max)
                st.write("Resultado de Max Pooling:")
                st.write(resultado)
                
        elif opcion == "Stride":
            st.write("Parámetros para Stride:")
            
            kernel_type = st.selectbox("Selecciona el tipo de kernel", 
                                        ["horizontal", "vertical"])
            tamaño_stride = st.number_input("Tamaño del Stride", min_value=1, value=2)
            
            if st.button("Calcular"):
                resultado = stride(matrix, kernel_type, tamaño_stride)
                st.write(f"Resultado de Stride con kernel {kernel_type} y stride {tamaño_stride}:")
                st.write(resultado)

        elif opcion == "Stacking":
            st.write("Parámetros para Stacking:")
            n_kernels = st.number_input("Número de kernels", min_value=1, value=2)

            if st.button("Calcular"):           
                resultados = stacking(matrix, n_kernels)
                for idx, resultado in enumerate(resultados):
                    st.write(f"Resultado de Stacking con Kernel {idx + 1}:")
                    st.write(resultado)
                    
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("Por favor, cargue un archivo válido.")