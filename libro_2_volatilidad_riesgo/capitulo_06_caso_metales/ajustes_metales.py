"""
ajustes_metales.py - Ajustes específicos y correlación para Metales (Oro y Plata)

Este módulo contiene funciones para manejar las diferencias contractuales 
entre el Oro y la Plata, así como para analizar su correlación y ajustar 
la volatilidad cruzada.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 6: Caso Práctico - Metales
"""

import numpy as np
import pandas as pd


# ============================================
# Especificaciones del Contrato (Broker estándar retail)
# ============================================
ESPECIFICACIONES_CONTRATO = {
    'XAUUSD': {
        'nombre': 'Oro',
        'tamano_lote': 100,       # 100 onzas por lote estándar
        'valor_tick': 0.01,       # Movimiento mínimo de precio
        'valor_pip': 0.1,         # Valor de un pip (0.10$ de movimiento = 10$ por lote)
        'margen_requerido': 0.05  # 5% de margen (apalancamiento 1:20)
    },
    'XAGUSD': {
        'nombre': 'Plata',
        'tamano_lote': 5000,      # 5000 onzas por lote estándar (o 100 en mini-lotes)
        'valor_tick': 0.005,      # Movimiento mínimo
        'valor_pip': 0.01,        # Valor de un pip (0.01$ de movimiento = 1$ por mini-lote)
        'margen_requerido': 0.10  # 10% de margen (apalancamiento 1:10, la plata pide más margen)
    }
}


def calcular_valor_pip(simbolo, lotes, movimiento_precio):
    """
    Calcula el beneficio o pérdida monetaria dado un movimiento de precio.
    
    Args:
        simbolo (str): 'XAUUSD' o 'XAGUSD'.
        lotes (float): Número de lotes operados.
        movimiento_precio (float): Diferencia de precio (Precio Salida - Precio Entrada).
    
    Returns:
        float: Beneficio/Pérdida en dólares.
    """
    specs = ESPECIFICACIONES_CONTRATO.get(simbolo)
    if not specs:
        raise ValueError(f"Símbolo {simbolo} no encontrado en las especificaciones.")
        
    # Fórmula general: PnL = Movimiento * Tamaño Lote * Lotes
    pnl = movimiento_precio * specs['tamano_lote'] * lotes
    return pnl


def calcular_correlacion_rodante(df_oro, df_plata, ventana=60):
    """
    Calcula la correlación rodante (rolling correlation) entre los rendimientos 
    del Oro y la Plata.
    
    Args:
        df_oro (pd.DataFrame): DataFrame con columna 'close' del Oro.
        df_plata (pd.DataFrame): DataFrame con columna 'close' de la Plata.
        ventana (int): Ventana de tiempo para calcular la correlación.
    
    Returns:
        pd.Series: Serie con la correlación rodante.
    """
    # Calcular log-retornos
    ret_oro = np.log(df_oro['close'] / df_oro['close'].shift(1))
    ret_plata = np.log(df_plata['close'] / df_plata['close'].shift(1))
    
    # Unir en un solo DataFrame para calcular la correlación
    df_retornos = pd.DataFrame({'oro': ret_oro, 'plata': ret_plata}).dropna()
    
    # Correlación rodante
    correlacion = df_retornos['oro'].rolling(ventana).corr(df_retornos['plata'])
    
    return correlacion


def ajustar_volatilidad_cruzada(df_oro, df_plata, objetivo_volatilidad=0.01):
    """
    Ajusta los rendimientos de ambos metales para que tengan la misma volatilidad 
    objetivo. Esto es útil para crear portafolios equilibrados (Risk Parity).
    
    Args:
        df_oro (pd.DataFrame): DataFrame con columna 'close' del Oro.
        df_plata (pd.DataFrame): DataFrame con columna 'close' de la Plata.
        objetivo_volatilidad (float): Volatilidad diaria objetivo (ej: 1%).
    
    Returns:
        dict: DataFrames con los rendimientos ajustados.
    """
    ret_oro = np.log(df_oro['close'] / df_oro['close'].shift(1)).dropna()
    ret_plata = np.log(df_plata['close'] / df_plata['close'].shift(1)).dropna()
    
    # Calcular volatilidad histórica (desviación estándar rodante de 20 días)
    vol_oro = ret_oro.rolling(20).std()
    vol_plata = ret_plata.rolling(20).std()
    
    # Factor de ajuste: (Volatilidad Objetivo / Volatilidad Actual)
    factor_oro = objetivo_volatilidad / vol_oro
    factor_plata = objetivo_volatilidad / vol_plata
    
    # Aplicar ajuste
    ret_oro_ajustado = ret_oro * factor_oro
    ret_plata_ajustado = ret_plata * factor_plata
    
    return {
        'oro_ajustado': ret_oro_ajustado,
        'plata_ajustada': ret_plata_ajustado,
        'factor_oro': factor_oro,
        'factor_plata': factor_plata
    }


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo ajustes_metales.py ===\n")
    
    # 1. Probar cálculo de PnL
    print("1. Cálculo de Beneficio/Pérdida:")
    pnl_oro = calcular_valor_pip('XAUUSD', lotes=1.0, movimiento_precio=10.5) # Oro sube 10.5$
    print(f"   Oro (1 lote, +10.5$): {pnl_oro:.2f}$")
    
    pnl_plata = calcular_valor_pip('XAGUSD', lotes=10.0, movimiento_precio=0.25) # Plata sube 0.25$
    print(f"   Plata (10 lotes, +0.25$): {pnl_plata:.2f}$\n")
    
    # 2. Probar Correlación y Ajuste de Volatilidad
    print("2. Correlación y Ajuste de Volatilidad:")
    np.random.seed(42)
    n = 200
    
    # Generar datos sintéticos correlacionados
    factor_comun = np.cumsum(np.random.randn(n) * 0.5)
    df_oro_test = pd.DataFrame({'close': 1800 + factor_comun + np.random.randn(n)*5})
    df_plata_test = pd.DataFrame({'close': 25 + (factor_comun * 0.02) + np.random.randn(n)*0.2})
    
    corr = calcular_correlacion_rodante(df_oro_test, df_plata_test, ventana=50)
    print(f"   Última correlación rodante (50 días): {corr.iloc[-1]:.3f}")
    
    ajustes = ajustar_volatilidad_cruzada(df_oro_test, df_plata_test)
    print(f"   Volatilidad ajustada Oro (último día): {ajustes['oro_ajustado'].iloc[-1]:.4f}")
    print(f"   Volatilidad ajustada Plata (último día): {ajustes['plata_ajustada'].iloc[-1]:.4f}")
    
    print("\n✓ Prueba completada exitosamente")
