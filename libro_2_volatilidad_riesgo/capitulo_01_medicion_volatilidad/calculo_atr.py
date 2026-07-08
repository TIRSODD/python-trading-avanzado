"""
calculo_atr.py - Cálculo del Average True Range (ATR)

Este módulo contiene funciones para calcular el ATR, 
una medida fundamental de volatilidad en trading.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 1: Medición y Modelado de Volatilidad
"""

import numpy as np
import pandas as pd


def calcular_true_range(df):
    """
    Calcula el True Range (TR) para cada vela.
    
    El True Range es el máximo de:
    1. High - Low (rango de la vela actual)
    2. |High - Close anterior| (gap alcista)
    3. |Low - Close anterior| (gap bajista)
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'high', 'low', 'close'
    
    Returns:
        pd.DataFrame: DataFrame con columna 'true_range' añadida
    
    Ejemplo:
        >>> df = pd.DataFrame({
        ...     'high': [100, 102, 101],
        ...     'low': [98, 100, 99],
        ...     'close': [99, 101, 100]
        ... })
        >>> df = calcular_true_range(df)
        >>> print(df['true_range'])
    """
    # Calcular los tres componentes del True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    # El True Range es el máximo de los tres
    df['true_range'] = np.maximum(high_low, np.maximum(high_close, low_close))
    
    return df


def calcular_atr(df, periodo=14):
    """
    Calcula el Average True Range (ATR).
    
    El ATR es la media móvil del True Range.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'high', 'low', 'close'
        periodo (int): Período de la media móvil (default: 14)
    
    Returns:
        pd.DataFrame: DataFrame con columnas 'atr' y 'atr_pct' añadidas
    
    Ejemplo:
        >>> df = calcular_atr(df, periodo=14)
        >>> print(f"ATR actual: {df['atr'].iloc[-1]:.4f}")
        >>> print(f"ATR%: {df['atr_pct'].iloc[-1]:.4f}%")
    """
    # Primero calcular el True Range
    df = calcular_true_range(df)
    
    # Calcular ATR como media móvil del True Range
    df['atr'] = df['true_range'].rolling(periodo).mean()
    
    # Calcular ATR como porcentaje del precio (para comparar entre activos)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df


def analizar_atr_normalizado(df, periodo=14):
    """
    Analiza el ATR normalizado para comparar volatilidad.
    
    Args:
        df (pd.DataFrame): DataFrame con datos OHLC
        periodo (int): Período para calcular ATR
    
    Returns:
        pd.DataFrame: DataFrame con ATR calculado
        dict: Estadísticas del ATR
    
    Ejemplo:
        >>> df, stats = analizar_atr_normalizado(df)
        >>> print(f"ATR% medio: {stats['medio']:.3f}%")
    """
    # Calcular ATR
    df = calcular_atr(df, periodo)
    
    # Obtener ATR% sin valores nulos
    atr_pct = df['atr_pct'].dropna()
    
    # Calcular estadísticas
    estadisticas = {
        'medio': atr_pct.mean(),
        'mediana': atr_pct.median(),
        'minimo': atr_pct.min(),
        'maximo': atr_pct.max(),
        'desviacion_std': atr_pct.std(),
        'percentil_10': atr_pct.quantile(0.10),
        'percentil_25': atr_pct.quantile(0.25),
        'percentil_50': atr_pct.quantile(0.50),
        'percentil_75': atr_pct.quantile(0.75),
        'percentil_90': atr_pct.quantile(0.90)
    }
    
    # Imprimir resultados
    print("=== ANÁLISIS DE ATR NORMALIZADO ===\n")
    print(f"ATR% medio: {estadisticas['medio']:.3f}%")
    print(f"ATR% mediana: {estadisticas['mediana']:.3f}%")
    print(f"ATR% mínimo: {estadisticas['minimo']:.3f}%")
    print(f"ATR% máximo: {estadisticas['maximo']:.3f}%")
    print(f"ATR% desviación estándar: {estadisticas['desviacion_std']:.3f}%\n")
    
    print("Percentiles de ATR%:")
    for p in [10, 25, 50, 75, 90]:
        valor = estadisticas[f'percentil_{p}']
        print(f"  P{p}: {valor:.3f}%")
    
    return df, estadisticas


# ============================================
# Código de prueba (solo se ejecuta si se corre directamente)
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo calculo_atr.py ===\n")
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n = 100
    
    df_ejemplo = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n)),
        'high': 100 + np.cumsum(np.random.randn(n)) + np.random.uniform(0, 2, n),
        'low': 100 + np.cumsum(np.random.randn(n)) - np.random.uniform(0, 2, n),
        'close': 100 + np.cumsum(np.random.randn(n))
    })
    
    # Calcular ATR
    df_resultado, stats = analizar_atr_normalizado(df_ejemplo, periodo=14)
    
    print("\n✓ Prueba completada exitosamente")
    print(f"\nÚltimos valores:")
    print(f"  Precio: {df_resultado['close'].iloc[-1]:.2f}")
    print(f"  ATR: {df_resultado['atr'].iloc[-1]:.4f}")
    print(f"  ATR%: {df_resultado['atr_pct'].iloc[-1]:.4f}%")
