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
            return 'baja'
        else:
            return 'normal'
            
    df['regimen_bollinger'] = df.apply(clasificar_bollinger, axis=1)
    
    return df


def detectar_regimenes_clustering(df, periodo_atr=14, num_clusters=3):
    """
    Detecta regímenes usando Machine Learning (K-Means) sobre múltiples 
    indicadores de volatilidad.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC y 'atr'.
        periodo_atr (int): Período del ATR.
        num_clusters (int): Número de regímenes a buscar (default: 3).
    
    Returns:
        pd.DataFrame: DataFrame con columna 'regimen_cluster' añadida.
    """
    # Preparar características (features) para el clustering
    df_features = df[['high', 'low', 'close', 'atr']].copy().dropna()
    
    # Calcular volatilidad de corto y largo plazo
    df_features['log_ret'] = np.log(df_features['close'] / df_features['close'].shift(1))
    df_features['vol_corta'] = df_features['log_ret'].rolling(5).std()
    df_features['vol_larga'] = df_features['log_ret'].rolling(20).std()
    df_features['rango'] = (df_features['high'] - df_features['low']) / df_features['close']
    
    # Seleccionar solo las columnas que usaremos para agrupar
    columnas_ml = ['atr', 'vol_corta', 'vol_larga', 'rango']
    datos_ml = df_features[columnas_ml].dropna()
    
    if len(datos_ml) < num_clusters:
        raise ValueError("No hay suficientes datos para realizar el clustering.")
    
    # Normalizar los datos (muy importante para K-Means)
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(datos_ml)
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_features.loc[datos_ml.index, 'cluster'] = kmeans.fit_predict(datos_normalizados)
    
    # Asignar etiquetas cualitativas (baja, normal, alta) basadas en el ATR medio de cada cluster
    cluster_stats = df_features.groupby('cluster')['atr'].mean()
    clusters_ordenados = cluster_stats.sort_values().index.tolist()
    
    etiquetas = {
        clusters_ordenados[0]: 'baja',
        clusters_ordenados[1]: 'normal',
        clusters_ordenados[2]: 'alta' if num_clusters > 2 else 'normal'
    }
    
    df_features['regimen_cluster'] = df_features['cluster'].map(etiquetas)
    
    # Unir resultados al DataFrame original
    df = df.join(df_features[['regimen_cluster']])
    
    return df


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo deteccion_regimenes.py ===\n")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n = 200
    
    df_ejemplo = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n)),
        'high': 100 + np.cumsum(np.random.randn(n)) + np.random.uniform(0, 2, n),
        'low': 100 + np.cumsum(np.random.randn(n)) - np.random.uniform(0, 2, n),
        'close': 100 + np.cumsum(np.random.randn(n))
    })
    
    # Simular ATR y ATR% (en un caso real usaríamos calcular_atr)
    df_ejemplo['atr'] = np.abs(np.random.randn(n)) * 2 + 1
    df_ejemplo['atr_pct'] = (df_ejemplo['atr'] / df_ejemplo['close']) * 100
    
    # Probar Método 1: Percentiles
    print("1. Detectando regímenes por percentiles...")
    df_res1 = detectar_regimenes_atr(df_ejemplo)
    print(df_res1['regimen_volatilidad'].value_counts())
    
    # Probar Método 2: Bollinger
    print("\n2. Detectando regímenes por Bandas de Bollinger...")
    df_res2 = detectar_regimenes_bollinger_atr(df_ejemplo)
    print(df_res2['regimen_bollinger'].value_counts())
    
    # Probar Método 3: Clustering
    print("\n3. Detectando regímenes por Clustering (K-Means)...")
    df_res3 = detectar_regimenes_clustering(df_ejemplo)
    print(df_res3['regimen_cluster'].value_counts())
    
    print("\n✓ Prueba completada exitosamente")
