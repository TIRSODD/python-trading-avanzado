"""
analisis_correlacion.py - Análisis de Correlación y Diversificación

Este módulo analiza la correlación entre los diferentes activos o estrategias 
del portafolio. El objetivo es asegurar que no estemos concentrando el riesgo 
en movimientos de mercado idénticos.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 7: Construcción de un Portafolio Multi-Estrategia
"""

import numpy as np
import pandas as pd


def calcular_matriz_correlacion(df_retornos):
    """
    Calcula la matriz de correlación de Pearson entre los retornos de las estrategias.
    
    Args:
        df_retornos (pd.DataFrame): DataFrame donde cada columna es una estrategia/activo.
    
    Returns:
        pd.DataFrame: Matriz de correlación.
    """
    return df_retornos.corr()


def calcular_correlacion_rodante(df_retornos, ventana=60):
    """
    Calcula cómo evoluciona la correlación en el tiempo.
    Las correlaciones no son estáticas; en momentos de pánico, todo tiende a correlacionarse a 1.
    
    Args:
        df_retornos (pd.DataFrame): DataFrame de retornos.
        ventana (int): Número de días para la media móvil.
    
    Returns:
        pd.DataFrame: Correlaciones rodantes.
    """
    return df_retornos.rolling(ventana).corr()


def evaluar_diversificacion(matriz_corr):
    """
    Evalúa la calidad de la diversificación del portafolio.
    
    Reglas generales:
    - Correlación > 0.7: Alta correlación (Poca diversificación).
    - Correlación entre 0.3 y 0.7: Correlación moderada.
    - Correlación < 0.3: Baja correlación (Buena diversificación).
    - Correlación negativa: Diversificación perfecta (se cubren las espaldas).
    """
    print("\n=== ANÁLISIS DE DIVERSIFICACIÓN ===\n")
    
    # Extraer solo la parte triangular inferior para no repetir pares
    n = len(matriz_corr)
    pares = []
    
    for i in range(n):
        for j in range(i + 1, n):
            nombre1 = matriz_corr.index[i]
            nombre2 = matriz_corr.columns[j]
            corr = matriz_corr.iloc[i, j]
            pares.append((nombre1, nombre2, corr))
            
    if not pares:
        print("No hay suficientes activos para analizar.")
        return
        
    print(f"{'Estrategia 1':<15} | {'Estrategia 2':<15} | {'Correlación':<12} | {'Valoración'}")
    print("-" * 65)
    
    for n1, n2, corr in pares:
        if corr > 0.7:
            valoracion = "⚠️ ALTA (Riesgo)"
        elif corr > 0.3:
            valoracion = "✅ MODERADA"
        elif corr > -0.3:
            valoracion = "🌟 BAJA (Excelente)"
        else:
            valoracion = "🏆 NEGATIVA (Cobertura perfecta)"
            
        print(f"{n1:<15} | {n2:<15} | {corr:<12.3f} | {valoracion}")


def sugerir_ajuste_pesos(matriz_corr, pesos_actuales):
    """
    Sugiere un ajuste simple: reducir peso a los activos muy correlacionados.
    (Lógica simplificada para fines educativos).
    """
    print("\n--- Sugerencia de Ajuste ---")
    # En un sistema real, aquí aplicaríamos optimización de media-varianza (Markowitz)
    # Para este ejemplo, solo imprimimos un consejo basado en la correlación media
    correlaciones = matriz_corr.values[np.triu_indices_from(matriz_corr, k=1)]
    media_corr = np.mean(correlaciones)
    
    print(f"Correlación media del portafolio: {media_corr:.3f}")
    if media_corr > 0.5:
        print("⚠️ Consejo: Tu portafolio está muy correlacionado. Considera añadir estrategias de activos diferentes (ej: Cripto, Materias Primas agrícolas).")
    else:
        print("✓ Consejo: El nivel de diversificación es aceptable.")


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo analisis_correlacion.py ===\n")
    
    np.random.seed(42)
    n = 252
    
    # Generar retornos sintéticos
    # EURGBP y ORO con correlación moderada
    factor_comun = np.random.normal(0, 1, n)
    
    ret_eurgbp = np.random.normal(0.0005, 0.005, n) + (factor_comun * 0.2)
    ret_oro = np.random.normal(0.001, 0.015, n) + (factor_comun * 0.5)
    # PLATA con correlación muy alta con el ORO
    ret_plata = ret_oro * 0.8 + np.random.normal(0, 0.005, n)
    
    df_retornos = pd.DataFrame({
        'EURGBP': ret_eurgbp,
        'ORO': ret_oro,
        'PLATA': ret_plata
    })
    
    # 1. Calcular matriz
    matriz = calcular_matriz_correlacion(df_retornos)
    print("Matriz de Correlación:")
    print(matriz.round(3))
    
    # 2. Evaluar diversificación
    evaluar_diversificacion(matriz)
    
    # 3. Sugerencia
    sugerir_ajuste_pesos(matriz, pesos_actuales=None)
    
    print("\n✓ Prueba completada exitosamente")
