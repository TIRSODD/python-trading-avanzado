"""
deteccion_regimenes.py - Detección de regímenes de volatilidad

Este módulo contiene funciones para identificar si el mercado está 
en un régimen de baja, normal o alta volatilidad.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 1: Medición y Modelado de Volatilidad
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Importamos la función calcular_atr del archivo anterior
# (En un entorno real, se importaría así: from calculo_atr import calcular_atr)
# Para este ejemplo, asumimos que el DataFrame ya tiene la columna 'atr' y 'atr_pct'


def detectar_regimenes_atr(df, periodo=14):
    """
    Detecta regímenes de volatilidad usando umbrales percentiles sobre el ATR%.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'atr_pct'.
        periodo (int): No usado aquí, pero mantenido por consistencia.
    
    Returns:
        pd.DataFrame: DataFrame con columna 'regimen_volatilidad' añadida.
    """
    # Asegurarnos de que tenemos datos
    if 'atr_pct' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'atr_pct'.")
        
    atr_pct = df['atr_pct'].dropna()
    
    # Definir umbrales (Percentil 25 y 75)
    umbral_bajo = atr_pct.quantile(0.25)
    umbral_alto = atr_pct.quantile(0.75)
    
    # Función para clasificar
    def clasificar_regimen(valor):
        if pd.isna(valor):
            return 'desconocido'
        elif valor < umbral_bajo:
            return 'baja'
        elif valor > umbral_alto:
            return 'alta'
        else:
            return 'normal'
            
    df['regimen_volatilidad'] = df['atr_pct'].apply(clasificar_regimen)
    
    return df


def detectar_regimenes_bollinger_atr(df, periodo_atr=14, periodo_bollinger=50, num_std=2):
    """
    Detecta regímenes usando Bandas de Bollinger aplicadas sobre el propio ATR.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'atr'.
        periodo_atr (int): Período del ATR.
        periodo_bollinger (int): Período de la media móvil para las bandas.
        num_std (int): Número de desviaciones estándar para las bandas.
    
    Returns:
        pd.DataFrame: DataFrame con columnas de bandas y régimen añadidas.
    """
    if 'atr' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'atr'.")
        
    # Calcular Bandas de Bollinger sobre el ATR
    df['atr_ma'] = df['atr'].rolling(periodo_bollinger).mean()
    df['atr_std'] = df['atr'].rolling(periodo_bollinger).std()
    
    df['atr_superior'] = df['atr_ma'] + (num_std * df['atr_std'])
    df['atr_inferior'] = df['atr_ma'] - (num_std * df['atr_std'])
    
    # Clasificar regímenes
    def clasificar_bollinger(row):
        if pd.isna(row['atr_superior']):
            return 'desconocido'
        elif row['atr'] > row['atr_superior']:
            return 'alta'
        elif row['atr'] < row['atr_inferior']:
            return '
