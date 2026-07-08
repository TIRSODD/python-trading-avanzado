"""
monte_carlo.py - Simulación de Monte Carlo para Backtesting

Este módulo utiliza simulaciones de Monte Carlo para probar la robustez 
de una estrategia. Reordena los retornos o las operaciones aleatoriamente 
para ver qué habría pasado en escenarios alternativos.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 4: Backtesting Robusto
"""

import numpy as np
import pandas as pd


def simulacion_monte_carlo_retornos(retornos_diarios, num_simulaciones=1000):
    """
    Realiza una simulación de Monte Carlo reordenando los retornos diarios.
    
    Esto responde a la pregunta: "¿Qué habría pasado si las ganancias y 
    pérdidas hubieran ocurrido en un orden diferente?"
    
    Args:
        retornos_diarios (list o pd.Series): Serie de retornos diarios.
        num_simulaciones (int): Número de simulaciones a realizar.
    
    Returns:
        np.ndarray: Array 2D con las curvas de equity simuladas.
    """
    retornos = np.array(retornos_diarios)
    curvas_equity = np.zeros((num_simulaciones, len(retornos)))
    
    for i in range(num_simulaciones):
        # Reordenar los retornos aleatoriamente
        retornos_aleatorios = np.random.permutation(retornos)
        # Calcular la curva de equity acumulada
        curvas_equity[i] = np.cumprod(1 + retornos_aleatorios)
        
    return curvas_equity


def simulacion_monte_carlo_trades(pnl_trades, num_simulaciones=1000):
    """
    Realiza una simulación de Monte Carlo reordenando los resultados de las operaciones.
    
    Args:
        pnl_trades (list o pd.Series): PnL (Profit and Loss) de cada operación.
        num_simulaciones (int): Número de simulaciones a realizar.
    
    Returns:
        np.ndarray: Array 2D con las curvas de equity de las operaciones simuladas.
    """
    trades = np.array(pnl_trades)
    curvas_equity = np.zeros((num_simulaciones, len(trades)))
    
    for i in range(num_simulaciones):
        trades_aleatorios = np.random.permutation(trades)
        curvas_equity[i] = np.cumsum(trades_aleatorios)
        
    return curvas_equity


def calcular_metricas_monte_carlo(curvas_equity, confianza=0.95):
    """
    Calcula las métricas de riesgo y rendimiento basadas en las simulaciones.
    
    Args:
        curvas_equity (np.ndarray): Array 2D con las curvas simuladas.
        confianza (float): Nivel de confianza para los intervalos (ej: 0.95 para 95%).
    
    Returns:
        dict: Diccionario con las métricas (peor drawdown, mejor retorno, etc.).
    """
    # Calcular métricas para cada simulación
    retornos_finales = curvas_equity[:, -1]
    
    # Calcular Drawdown máximo para cada curva
    drawdowns = []
    for curva in curvas_equity:
        pico_acumulado = np.maximum.accumulate(curva)
        drawdown = (curva - pico_acumulado) / pico_acumulado
        drawdowns.append(np.min(drawdown))
        
    drawdowns = np.array(drawdowns)
    
    # Calcular percentiles
    alpha = 1 - confianza
    percentil_bajo = alpha / 2
    percentil_alto = 1 - percentil_bajo
    
    metricas = {
        'retorno_medio': np.mean(retornos_finales),
        'retorno_percentil_5': np.percentile(retornos_finales, percentil_bajo * 100),
        'retorno_percentil_95': np.percentile(retornos_finales, percentil_alto * 100),
        'drawdown_medio': np.mean(drawdowns),
        'peor_drawdown': np.min(drawdowns),
        'drawdown_percentil_95': np.percentile(drawdowns, percentil_alto * 100)
    }
    
    return metricas


def imprimir_informe_monte_carlo(metricas, equity_real_final, drawdown_real):
    """
    Imprime un informe comparando los resultados reales con los simulados.
    """
    print("=== INFORME MONTE CARLO ===\n")
    print(f"Retorno Final Real: {equity_real_final:.2f}")
    print(f"  -> Percentil 5% simulado: {metricas['retorno_percentil_5']:.2f}")
    print(f"  -> Percentil 95% simulado: {metricas['retorno_percentil_95']:.2f}\n")
    
    print(f"Drawdown Máximo Real: {drawdown_real:.2f}%")
    print(f"  -> Peor Drawdown simulado: {metricas['peor_drawdown']:.2f}%")
    print(f"  -> Percentil 95% Drawdown simulado: {metricas['drawdown_percentil_95']:.2f}%\n")
    
    # Análisis de robustez
    if equity_real_final < metricas['retorno_percentil_5']:
        print("⚠️ ALERTA: Tu rendimiento real es peor que el 95% de las simulaciones. Posible suerte en el backtest.")
    elif drawdown_real < metricas['peor_drawdown']:
        print("⚠️ ALERTA: Tu drawdown real es mejor que el peor caso simulado. Cuidado con el overfitting.")
    else:
        print("✓ La estrategia parece robusta frente a la aleatoriedad del mercado.")


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo monte_carlo.py ===\n")
    
    # 1. Simulación con retornos diarios
    print("1. Simulación basada en retornos diarios:")
    np.random.seed(42)
    retornos_sinteticos = np.random.normal(0.001, 0.02, 252) # 1 año de datos
    
    curvas_ret = simulacion_monte_carlo_retornos(retornos_sinteticos, num_simulaciones=500)
    metricas_ret = calcular_metricas_monte_carlo(curvas_ret)
    
    equity_real = np.cumprod(1 + retornos_sinteticos)[-1]
    imprimir_informe_monte_carlo(metricas_ret, equity_real, -15.0)
    
    print("\n" + "="*40 + "\n")
    
    # 2. Simulación con PnL de operaciones
    print("2. Simulación basada en operaciones (trades):")
    # Simular 100 operaciones con sesgo alcista
    trades_sinteticos = np.random.normal(50, 150, 100) 
    
    curvas_trades = simulacion_monte_carlo_trades(trades_sinteticos, num_simulaciones=500)
    metricas_trades = calcular_metricas_monte_carlo(curvas_trades)
    
    equity_trades_real = np.cumsum(trades_sinteticos)[-1]
    imprimir_informe_monte_carlo(metricas_trades, equity_trades_real, -1200.0)
    
    print("\n✓ Prueba completada exitosamente")
