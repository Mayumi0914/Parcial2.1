import pandas as pd
import numpy as np

horizontal_kernel = np.array([[1, 1, 1],
                              [ 0,  0,  0],
                              [-1, -1, -1]])

vertical_kernel = np.array([[1,  0, -1],
                            [1,  0, -1],
                            [1,  0, -1]])

def convolution(matrix, kernel_type='horizontal'):

    if kernel_type == 'horizontal':
        kernel = horizontal_kernel
    elif kernel_type == 'vertical':
        kernel = vertical_kernel
    else: f"No es valido el kernel"

    filas, columnas = matrix.shape
    salida = np.zeros((filas - 2, columnas - 2))
    
    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            region = matrix[i-1:i+2, j-1:j+2]
            salida[i-1, j-1] = np.sum(region * kernel)
    
    return salida

def generar_kernels(n):
    kernels = []
    for _ in range(n):
        kernel = np.random.rand(3, 3) * 2 - 1  
        kernels.append(kernel)
    return kernels
    
def convolution2(matrix, kernel):
    
    filas, columnas = matrix.shape
    salida = np.zeros((filas - 2, columnas - 2))

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            region = matrix[i-1:i+2, j-1:j+2]
            salida[i-1, j-1] = np.sum(region * kernel)
    
    return salida

def stacking(matrix, n):
    
    kernels = generar_kernels(n)
    
    resultados = []
    
    for kernel in kernels:
        resultado = convolution2(matrix, kernel)
        resultados.append(resultado)
    
    return resultados


def stride(matrix, kernel_type='horizontal', stride=2):
    if kernel_type == 'horizontal':
        kernel = horizontal_kernel
    elif kernel_type == 'vertical':
        kernel = vertical_kernel
    else:f"No es valido el kernel"

    filas, columnas = matrix.shape
    
    nueva_fila = (filas - 2) // stride
    nueva_col = (columnas - 2) // stride

    salida = np.zeros((nueva_fila , nueva_col))
    
    for i in range(0, filas - 2, stride):
        for j in range(0, columnas - 2, stride):
            region = matrix[i:i+3, j:j+3]
            salida[i // stride, j // stride] = np.sum(region * kernel)
    
    return salida


def padding(matrix, p_filas, p_columnas=0):

    filas, columnas = matrix.shape

    nuevas_filas = filas + 2 * p_filas
    nuevas_columnas = columnas + 2 * p_columnas
    
    salida = np.zeros((nuevas_filas, nuevas_columnas))
    
    salida[p_filas:p_filas + filas, p_columnas:p_columnas + columnas] = matrix
    
    return salida

def max_pooling(matrix, stride):
    filas, columnas = matrix.shape

    nueva_fila = (filas - 1) // stride +1
    nueva_col = (columnas - 1) // stride + 1

    salida = np.zeros((nueva_fila, nueva_col))
    
    for i in range(nueva_fila):
        for j in range(nueva_col):
            start_i = i * stride
            start_j = j * stride
            
            b = matrix[start_i:start_i+2, start_j:start_j+2]
            salida[i, j] = np.max(b)
    
    return salida
