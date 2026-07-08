"""
asignacion_capital.py - Métodos de Asignación de Capital (Weighting)

Este módulo implementa diferentes métodos para decidir cuánto capital 
asignar a cada estrategia dentro de un portafolio multi-estrategia.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 7: Construcción de un Portafolio Multi-Estrategia
"""

import numpy as np
import pandas as pd


def pesos_iguales(n_activos):
    """
    Asignación ingenua: mismo capital a todas las estrategias.
    Es el punto de partida (benchmark) para comparar otros métodos.
    """
    return np.array([1.0 / n_activos] * n_activos)


def pesos_inversa_volatilidad(retornos):
    """
    Asigna más capital a las estrategias menos volátiles y menos a las más arriesgadas.
    Muy usado para proteger la cuenta en momentos de mercado nervioso.
    """
    # Calcular volatilidad histórica (desviación estándar)
    volatilidades = retornos.std()
    
    # Si alguna volatilidad es 0, evitar división por cero
    volatilidades = volatilidades.replace(0, np.nan)
    
    # Calcular pesos inversos
    pesos_inversos = 1.0 / volatilidades
    pesos_inversos = pesos_inversos.dropna()
    
    # Normalizar para que la suma de todos los pesos sea 1 (100%)
    pesos_finales = pesos_inversos / pesos_inversos.sum()
    
    return pesos_finales


def pesos_sharpe_ratio(retornos, tasa_libre_riesgo=0.0):
    """
    Asigna más capital a las estrategias que históricamente han tenido 
    un mejor Ratio de Sharpe (mejor rentabilidad ajustada al riesgo).
    """
    # Calcular Sharpe de cada estrategia
    sharpe = (retornos.mean() - tasa_libre_riesgo) / retornos.std()
    
    # Solo considerar estrategias con Sharpe positivo (ignoramos las que pierden)
    sharpe_positivo = sharpe.clip(lower=0)
    
    # Si todas son negativas, repartir por igual para no concentrar en una mala
    if sharpe_positivo.sum() == 0:
        return pesos_iguales(len(retornos.columns))
        
    # Normalizar los pesos
    pesos_finales = sharpe_positivo / sharpe_positivo.sum()
    return pesos_finales


def imprimir_tabla_pesos(df_pesos):
    """
    Imprime una tabla comparativa de los métodos de asignación.
    """
    print("\n=== COMPARATIVA DE ASIGNACIÓN DE CAPITAL ===\n")
    print(df_pesos.round(4).to_string(index=False))


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo asignacion_capital.py ===\n")
    
    np.random.seed(42)
    n = 252 # 1 año de datos diarios
    
    # Simular retornos de 3 estrategias con perfiles diferentes
    # Estrategia A: Estable, baja volatilidad (ej: EUR/GBP)
    ret_a = np.random.normal(0.0005, 0.005, n)
    # Estrategia B: Más rentable pero más volátil (ej: Oro)
    ret_b = np.random.normal(0.0010, 0.015, n)
    # Estrategia C: Mala estrategia (Sharpe negativo, perdiendo dinero)
    ret_c = np.random.normal(-0.0002, 0.010, n)
    
    df_retornos = pd.DataFrame({
        'EURGBP': ret_a,
        'ORO': ret_b,
        'PLATA': ret_c
    })
    
    # 1. Calcular Pesos Iguales (33.3% cada una)
    p_iguales = pesos_iguales(3)
    
    # 2. Calcular Inversa Volatilidad
    p_inv_vol = pesos_inversa_volatilidad(df_retornos)
    
    # 3. Calcular Sharpe Ratio
    p_sharpe = pesos_sharpe_ratio(df_retornos)
    
    # Crear tabla para imprimir
    tabla = pd.DataFrame({
        'Estrategia': df_retornos.columns,
        'Pesos Iguales (%)': p_iguales * 100,
        'Inversa Volatilidad (%)': p_inv_vol.values * 100,
        'Sharpe Ratio (%)': p_sharpe.values * 100
    })
    
    imprimir_tabla_pesos(tabla)
    
    print("\n✓ Prueba completada exitosamente")
