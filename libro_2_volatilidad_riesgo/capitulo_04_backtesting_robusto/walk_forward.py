"""
walk_forward.py - Análisis Walk-Forward para validación de estrategias

Este módulo implementa el análisis Walk-Forward, una técnica avanzada
de backtesting que divide los datos en ventanas de optimización y 
validación para detectar overfitting.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 4: Backtesting Robusto
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_simple(df, estrategia_func, parametros_range, 
                        ventana_train=100, ventana_test=50):
    """
    Implementa un análisis Walk-Forward simple.
    
    Divide los datos en ventanas consecutivas:
    - Ventana de entrenamiento: para optimizar parámetros
    - Ventana de prueba: para validar con parámetros óptimos
    
    Args:
        df (pd.DataFrame): DataFrame con datos OHLC.
        estrategia_func (function): Función de la estrategia que acepta df y parámetros.
        parametros_range (dict): Diccionario con rangos de parámetros a probar.
        ventana_train (int): Tamaño de la ventana de entrenamiento.
        ventana_test (int): Tamaño de la ventana de prueba.
    
    Returns:
        pd.DataFrame: Resultados de cada ventana walk-forward.
    """
    resultados = []
    total_ventanas = (len(df) - ventana_train) // ventana_test
    
    print(f"=== ANÁLISIS WALK-FORWARD ===")
    print(f"Total ventanas: {total_ventanas}")
    print(f"Ventana train: {ventana_train}, Ventana test: {ventana_test}\n")
    
    for i in range(total_ventanas):
        # Definir índices de las ventanas
        inicio_train = i * ventana_test
        fin_train = inicio_train + ventana_train
        fin_test = fin_train + ventana_test
        
        # Extraer datos de entrenamiento y prueba
        df_train = df.iloc[inicio_train:fin_train]
        df_test = df.iloc[fin_train:fin_test]
        
        # Optimizar parámetros en ventana de entrenamiento
        mejor_param, mejor_sharpe = optimizar_parametros(
            df_train, estrategia_func, parametros_range
        )
        
        # Validar en ventana de prueba
        resultado_test = validar_estrategia(
            df_test, estrategia_func, mejor_param
        )
        
        resultados.append({
            'ventana': i + 1,
            'periodo_train': f"{inicio_train}-{fin_train}",
            'periodo_test': f"{fin_train}-{fin_test}",
            'mejores_parametros': mejor_param,
            'sharpe_train': mejor_sharpe,
            'sharpe_test': resultado_test['sharpe'],
            'profit_factor_test': resultado_test['profit_factor'],
            'operaciones_test': resultado_test['num_operaciones']
        })
        
        print(f"Ventana {i+1}: Sharpe Train={mejor_sharpe:.2f}, Sharpe Test={resultado_test['sharpe']:.2f}")
    
    return pd.DataFrame(resultados)


def optimizar_parametros(df, estrategia_func, parametros_range):
    """
    Optimiza los parámetros de una estrategia en un conjunto de datos.
    
    Args:
        df (pd.DataFrame): Datos de entrenamiento.
        estrategia_func (function): Función de la estrategia.
        parametros_range (dict): Rangos de parámetros a probar.
    
    Returns:
        tuple: (mejores_parametros, mejor_sharpe)
    """
    mejor_sharpe = -999
    mejores_params = None
    
    # Generar todas las combinaciones de parámetros
    from itertools import product
    
    nombres_params = list(parametros_range.keys())
    valores_params = list(parametros_range.values())
    combinaciones = list(product(*valores_params))
    
    for combinacion in combinaciones:
        params = dict(zip(nombres_params, combinacion))
        
        # Ejecutar estrategia con estos parámetros
        try:
            resultado = estrategia_func(df, **params)
            sharpe = calcular_sharpe(resultado)
            
            if sharpe > mejor_sharpe:
                mejor_sharpe = sharpe
                mejores_params = params
        except Exception:
            continue
    
    return mejores_params, mejor_sharpe


def validar_estrategia(df, estrategia_func, parametros):
    """
    Valida una estrategia con parámetros fijos en datos de prueba.
    
    Args:
        df (pd.DataFrame): Datos de prueba.
        estrategia_func (function): Función de la estrategia.
        parametros (dict): Parámetros a usar.
    
    Returns:
        dict: Métricas de rendimiento.
    """
    resultado = estrategia_func(df, **parametros)
    
    return {
        'sharpe': calcular_sharpe(resultado),
        'profit_factor': calcular_profit_factor(resultado),
        'num_operaciones': len(resultado) if isinstance(resultado, list) else 0
    }


def calcular_sharpe(retornos, tasa_libre_riesgo=0.02):
    """Calcula el Ratio de Sharpe anualizado."""
    if len(retornos) < 2:
        return 0.0
    
    retornos = pd.Series(retornos)
    exceso_retorno = retornos.mean() - (tasa_libre_riesgo / 252)
    sharpe = exceso_retorno / retornos.std() * np.sqrt(252)
    
    return sharpe


def calcular_profit_factor(operaciones):
    """Calcula el Profit Factor."""
    if len(operaciones) == 0:
        return 0.0
    
    ganancias = sum([op for op in operaciones if op > 0])
    perdidas = abs(sum([op for op in operaciones if op < 0]))
    
    if perdidas == 0:
        return float('inf')
    
    return ganancias / perdidas


def analizar_eficiencia_walk_forward(df_resultados):
    """
    Analiza la eficiencia del modelo comparando Sharpe de train vs test.
    
    Una estrategia robusta debe tener Sharpe Test cercano al Sharpe Train.
    """
    if len(df_resultados) == 0:
        return None
    
    sharpe_train_medio = df_resultados['sharpe_train'].mean()
    sharpe_test_medio = df_resultados['sharpe_test'].mean()
    
    # Calcular ratio de eficiencia (cuánto del rendimiento de train se mantiene en test)
    if sharpe_train_medio > 0:
        eficiencia = (sharpe_test_medio / sharpe_train_medio) * 100
    else:
        eficiencia = 0
    
    print(f"\n=== ANÁLISIS DE EFICIENCIA ===")
    print(f"Sharpe Train medio: {sharpe_train_medio:.2f}")
    print(f"Sharpe Test medio:  {sharpe_test_medio:.2f}")
    print(f"Eficiencia: {eficiencia:.1f}%")
    
    if eficiencia > 70:
        print("✓ Excelente: La estrategia es robusta")
    elif eficiencia > 50:
        print(" Aceptable: La estrategia tiene cierto overfitting")
    else:
        print(" Pobre: La estrategia tiene overfitting significativo")
    
    return eficiencia


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo walk_forward.py ===\n")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n = 500
    
    df_ejemplo = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n))
    })
    
    # Función de estrategia simple (solo para prueba)
    def estrategia_simple(df, periodo_ma=20):
        """Estrategia de cruce de media móvil simple."""
        df['ma'] = df['close'].rolling(periodo_ma).mean()
        señales = []
        for i in range(periodo_ma, len(df)):
            if df['close'].iloc[i] > df['ma'].iloc[i] and df['close'].iloc[i-1] <= df['ma'].iloc[i-1]:
                señales.append(1)  # Compra
            elif df['close'].iloc[i] < df['ma'].iloc[i] and df['close'].iloc[i-1] >= df['ma'].iloc[i-1]:
                señales.append(-1)  # Venta
        return señales
    
    # Rangos de parámetros a probar
    parametros_range = {
        'periodo_ma': [10, 20, 30, 50]
    }
    
    # Ejecutar Walk-Forward
    print("1. Ejecutando análisis Walk-Forward...")
    resultados = walk_forward_simple(
        df_ejemplo, 
        estrategia_simple, 
        parametros_range,
        ventana_train=100,
        ventana_test=50
    )
    
    # Analizar eficiencia
    print("\n2. Analizando eficiencia...")
    eficiencia = analizar_eficiencia_walk_forward(resultados)
    
    print("\n✓ Prueba completada exitosamente")
