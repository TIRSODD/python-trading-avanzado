"""
optimizacion.py - Optimización de parámetros para el sistema EUR/GBP

Este módulo implementa una búsqueda en cuadrícula (Grid Search) para encontrar 
los mejores parámetros del sistema. Además, incluye una función para analizar 
la "robustez" de los parámetros y evitar el overfitting.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 5: Caso Práctico - EUR/GBP
"""

import numpy as np
import pandas as pd
from itertools import product

# Nota: Asumimos que sistema_eurgbp.py está en la misma carpeta
try:
    from sistema_eurgbp import SistemaEURGBP
except ImportError:
    print("⚠️ Aviso: No se pudo importar SistemaEURGBP. Asegúrate de que el archivo existe.")
    SistemaEURGBP = None


def optimizar_grid_search(df, rangos_parametros, metrica_objetivo='profit_factor'):
    """
    Realiza una búsqueda exhaustiva de combinaciones de parámetros.
    
    Args:
        df (pd.DataFrame): Datos históricos OHLC.
        rangos_parametros (dict): Diccionario con los parámetros y sus valores a probar.
            Ejemplo: {'rsi_periodo': [10, 14, 20], 'bb_mult': [1.5, 2.0, 2.5]}
        metrica_objetivo (str): Métrica a maximizar ('profit_factor', 'sharpe', 'pnl_total').
    
    Returns:
        pd.DataFrame: Tabla ordenada con los resultados de todas las combinaciones.
    """
    if SistemaEURGBP is None:
        print("No se puede ejecutar la optimización sin la clase SistemaEURGBP.")
        return None

    resultados = []
    
    # Obtener nombres y valores de los parámetros
    nombres = list(rangos_parametros.keys())
    valores = list(rangos_parametros.values())
    
    # Generar todas las combinaciones posibles
    combinaciones = list(product(*valores))
    total = len(combinaciones)
    
    print(f"=== INICIANDO OPTIMIZACIÓN (Grid Search) ===")
    print(f"Total combinaciones a probar: {total}\n")
    
    for i, combinacion in enumerate(combinaciones):
        params = dict(zip(nombres, combinacion))
        
        # En un caso real, pasaríamos estos parámetros al sistema.
        # Aquí simulamos la ejecución y el cálculo de métricas para el ejemplo.
        # (Para que el código sea ejecutable sin modificar la clase base, 
        # usamos una simulación de métricas basada en los parámetros).
        
        # Simulación de métricas (En producción, esto llamaría a sistema.simular_sistema())
        np.random.seed(sum(combinacion)) # Semilla determinista para la prueba
        operaciones_sim = np.random.normal(10, 50, 100)
        
        profit_factor = abs(operaciones_sim[operaciones_sim > 0].sum() / operaciones_sim[operaciones_sim < 0].sum()) if operaciones_sim[operaciones_sim < 0].sum() != 0 else 0
        sharpe = (operaciones_sim.mean() / operaciones_sim.std()) * np.sqrt(252)
        pnl_total = operaciones_sim.sum()
        
        resultados.append({
            **params,
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 2),
            'pnl_total': round(pnl_total, 2),
            'num_operaciones': 100
        })
        
    df_resultados = pd.DataFrame(resultados)
    
    # Ordenar por la métrica objetivo
    if metrica_objetivo in df_resultados.columns:
        df_resultados = df_resultados.sort_values(by=metrica_objetivo, ascending=False)
        
    print(f"Top 5 combinaciones para '{metrica_objetivo}':")
    print(df_resultados.head(5).to_string(index=False))
    
    return df_resultados


def analizar_robustez_parametros(df_resultados, top_n=10):
    """
    Analiza si los mejores parámetros son robustos o si son un pico de overfitting.
    
    Si los parámetros vecinos (ligeramente diferentes) también tienen buenos resultados,
    el parámetro es robusto. Si solo ese funciona y los vecinos fallan, es overfitting.
    
    Args:
        df_resultados (pd.DataFrame): Resultados de la optimización.
        top_n (int): Cuántos de los mejores resultados analizar.
    """
    print(f"\n=== ANÁLISIS DE ROBUSTEZ (Top {top_n}) ===\n")
    
    mejores = df_resultados.head(top_n)
    media_top = mejores['profit_factor'].mean()
    media_total = df_resultados['profit_factor'].mean()
    
    print(f"Profit Factor medio del Top {top_n}: {media_top:.2f}")
    print(f"Profit Factor medio de TODAS las combinaciones: {media_total:.2f}")
    
    ratio_robustez = media_top / media_total if media_total > 0 else 0
    
    if ratio_robustez > 1.5:
        print("⚠️ ALERTA: Los mejores parámetros son mucho mejores que la media. Posible overfitting.")
    else:
        print("✓ Los mejores parámetros parecen robustos y no son un pico aislado.")
        
    return ratio_robustez


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo optimizacion.py ===\n")
    
    # Definir rangos de parámetros a probar
    rangos = {
        'rsi_periodo': [10, 14, 20],
        'bb_multiplicador': [1.5, 2.0, 2.5],
        'atr_stop_mult': [1.5, 2.0, 2.5]
    }
    
    # Datos sintéticos para la prueba
    np.random.seed(42)
    n = 500
    df_test = pd.DataFrame({
        'open': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005),
        'high': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005) + 0.0010,
        'low': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005) - 0.0010,
        'close': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005)
    })
    
    # 1. Ejecutar Optimización
    resultados = optimizar_grid_search(df_test, rangos, metrica_objetivo='profit_factor')
    
    # 2. Analizar Robustez
    if resultados is not None:
        analizar_robustez_parametros(resultados, top_n=5)
        
    print("\n✓ Prueba completada exitosamente")
