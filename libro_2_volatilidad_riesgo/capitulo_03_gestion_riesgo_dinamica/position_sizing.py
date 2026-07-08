"""
position_sizing.py - Cálculo dinámico del tamaño de posición

Este módulo implementa diferentes métodos de position sizing basados 
en el riesgo, la volatilidad (ATR) y el capital disponible.
Es la base de la gestión de riesgo profesional.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 3: Gestión de Riesgo Dinámica
"""

import numpy as np
import pandas as pd


def calcular_tamano_posicion_fijo_riesgo(capital, riesgo_pct, stop_loss_pips, 
                                          valor_pip=10):
    """
    Calcula el tamaño de posición basado en un riesgo fijo porcentual.
    
    Fórmula: Tamaño = (Capital * Riesgo%) / (Stop en pips * Valor del pip)
    
    Args:
        capital (float): Capital total disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación (ej: 1.0 para 1%).
        stop_loss_pips (float): Distancia del stop loss en pips.
        valor_pip (float): Valor monetario de 1 pip por lote estándar (default: $10).
    
    Returns:
        float: Tamaño de posición en lotes.
    
    Ejemplo:
        >>> lotes = calcular_tamano_posicion_fijo_riesgo(10000, 1.0, 50)
        >>> print(f"Tamaño: {lotes} lotes")
        Tamaño: 0.2 lotes
    """
    if stop_loss_pips <= 0:
        raise ValueError("El stop loss debe ser mayor que 0.")
    
    riesgo_monetario = capital * (riesgo_pct / 100)
    tamaño_lotes = riesgo_monetario / (stop_loss_pips * valor_pip)
    
    return round(tamaño_lotes, 2)


def calcular_tamano_posicion_atr(capital, riesgo_pct, atr, multiplicador_atr=2.0,
                                  valor_pip=10):
    """
    Calcula el tamaño de posición basado en el ATR (Average True Range).
    
    El stop loss se define dinámicamente como: ATR * multiplicador.
    Esto ajusta el tamaño de la posición a la volatilidad actual del mercado.
    
    Args:
        capital (float): Capital total disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación.
        atr (float): Valor actual del ATR (en pips).
        multiplicador_atr (float): Multiplicador para calcular el stop (default: 2.0).
        valor_pip (float): Valor monetario de 1 pip por lote estándar.
    
    Returns:
        dict: Diccionario con tamaño_lotes, stop_loss_pips y riesgo_monetario.
    """
    # Calcular el stop loss en pips basado en la volatilidad actual
    stop_loss_pips = atr * multiplicador_atr
    
    if stop_loss_pips <= 0:
        raise ValueError("El stop loss calculado debe ser mayor que 0.")
    
    riesgo_monetario = capital * (riesgo_pct / 100)
    tamaño_lotes = riesgo_monetario / (stop_loss_pips * valor_pip)
    
    return {
        'tamaño_lotes': round(tamaño_lotes, 2),
        'stop_loss_pips': round(stop_loss_pips, 2),
        'riesgo_monetario': round(riesgo_monetario, 2)
    }


def calcular_tamano_posicion_volatilidad_ajustada(capital, riesgo_pct, atr_actual, 
                                                    atr_medio, multiplicador_atr=2.0,
                                                    valor_pip=10):
    """
    Ajusta el tamaño de posición según la volatilidad relativa.
    
    Si la volatilidad actual es mayor que la media, reduce el tamaño de la posición
    para mantener el riesgo constante. Si es menor, lo aumenta (hasta un límite seguro).
    
    Args:
        capital (float): Capital total disponible.
        riesgo_pct (float): Porcentaje de riesgo base.
        atr_actual (float): ATR actual del mercado.
        atr_medio (float): ATR medio histórico (referencia).
        multiplicador_atr (float): Multiplicador para el stop.
        valor_pip (float): Valor monetario de 1 pip.
    
    Returns:
        dict: Diccionario con tamaño ajustado y factor de ajuste.
    """
    # Calcular ratio de volatilidad (cuánto se desvía la volatilidad actual de la media)
    ratio_volatilidad = atr_actual / atr_medio
    
    # Ajustar el riesgo: si volatilidad es alta (ratio > 1), reducimos el riesgo.
    # Limitamos el ajuste entre 0.5x y 1.5x para no ser demasiado agresivos.
    factor_ajuste = 1.0 / ratio_volatilidad
    factor_ajuste = max(0.5, min(factor_ajuste, 1.5))  
    
    riesgo_ajustado = riesgo_pct * factor_ajuste
    
    # Calcular tamaño con el riesgo ajustado
    stop_loss_pips = atr_actual * multiplicador_atr
    riesgo_monetario = capital * (riesgo_ajustado / 100)
    tamaño_lotes = riesgo_monetario / (stop_loss_pips * valor_pip)
    
    return {
        'tamaño_lotes': round(tamaño_lotes, 2),
        'riesgo_ajustado_pct': round(riesgo_ajustado, 2),
        'factor_ajuste': round(factor_ajuste, 2),
        'stop_loss_pips': round(stop_loss_pips, 2)
    }


def comparar_metodos_position_sizing(capital, riesgo_pct, atr, atr_medio, 
                                      stop_fijo_pips=50):
    """
    Compara los diferentes métodos de position sizing en una tabla.
    
    Args:
        capital (float): Capital total.
        riesgo_pct (float): Porcentaje de riesgo.
        atr (float): ATR actual.
        atr_medio (float): ATR medio histórico.
        stop_fijo_pips (float): Stop fijo para el método tradicional.
    
    Returns:
        pd.DataFrame: Tabla comparativa de los métodos.
    """
    # Método 1: Riesgo fijo con stop fijo
    lotes_fijo = calcular_tamano_posicion_fijo_riesgo(capital, riesgo_pct, stop_fijo_pips)
    
    # Método 2: Basado en ATR
    resultado_atr = calcular_tamano_posicion_atr(capital, riesgo_pct, atr)
    
    # Método 3: Volatilidad ajustada
    resultado_ajustado = calcular_tamano_posicion_volatilidad_ajustada(
        capital, riesgo_pct, atr, atr_medio
    )
    
    # Crear tabla comparativa
    comparacion = pd.DataFrame({
        'Método': ['Riesgo Fijo', 'Basado en ATR', 'Volatilidad Ajustada'],
        'Tamaño (lotes)': [lotes_fijo, resultado_atr['tamaño_lotes'], 
                          resultado_ajustado['tamaño_lotes']],
        'Stop (pips)': [stop_fijo_pips, resultado_atr['stop_loss_pips'],
                       resultado_ajustado['stop_loss_pips']],
        'Riesgo ($)': [capital * riesgo_pct / 100, resultado_atr['riesgo_monetario'],
                      capital * resultado_ajustado['riesgo_ajustado_pct'] / 100]
    })
    
    print("=== COMPARACIÓN DE MÉTODOS DE POSITION SIZING ===\n")
    print(comparacion.to_string(index=False))
    
    return comparacion


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo position_sizing.py ===\n")
    
    # Parámetros de ejemplo
    capital_ejemplo = 10000
    riesgo_pct = 1.0
    atr_ejemplo = 50  # 50 pips de volatilidad actual
    atr_medio_ejemplo = 40  # 40 pips de volatilidad media histórica
    
    print("1. Método de riesgo fijo (Stop de 50 pips):")
    lotes = calcular_tamano_posicion_fijo_riesgo(capital_ejemplo, riesgo_pct, 50)
    print(f"   Tamaño: {lotes} lotes\n")
    
    print("2. Método basado en ATR (Volatilidad actual):")
    resultado = calcular_tamano_posicion_atr(capital_ejemplo, riesgo_pct, atr_ejemplo)
    print(f"   Tamaño: {resultado['tamaño_lotes']} lotes")
    print(f"   Stop: {resultado['stop_loss_pips']} pips")
    print(f"   Riesgo: ${resultado['riesgo_monetario']}\n")
    
    print("3. Método de volatilidad ajustada:")
    resultado_ajustado = calcular_tamano_posicion_volatilidad_ajustada(
        capital_ejemplo, riesgo_pct, atr_ejemplo, atr_medio_ejemplo
    )
    print(f"   Tamaño: {resultado_ajustado['tamaño_lotes']} lotes")
    print(f"   Riesgo ajustado: {resultado_ajustado['riesgo_ajustado_pct']}%")
    print(f"   Factor de ajuste: {resultado_ajustado['factor_ajuste']}x\n")
    
    print("4. Comparación de métodos:")
    comparar_metodos_position_sizing(capital_ejemplo, riesgo_pct, 
                                      atr_ejemplo, atr_medio_ejemplo)
    
    print("\n✓ Prueba completada exitosamente")
