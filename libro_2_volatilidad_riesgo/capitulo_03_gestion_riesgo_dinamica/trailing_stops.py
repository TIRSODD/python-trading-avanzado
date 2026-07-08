"""
trailing_stops.py - Gestión dinámica de Trailing Stops

Este módulo implementa diferentes tipos de Trailing Stops (stops dinámicos)
para proteger beneficios a medida que el precio avanza a nuestro favor.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 3: Gestión de Riesgo Dinámica
"""

import numpy as np
import pandas as pd


def trailing_stop_fijo(precios, distancia_pips, direccion='long', valor_pip=10):
    """
    Calcula un Trailing Stop basado en una distancia fija en pips.
    
    Args:
        precios (pd.Series o list): Serie de precios históricos.
        distancia_pips (float): Distancia del stop en pips.
        direccion (str): 'long' (compra) o 'short' (venta).
        valor_pip (float): Valor monetario de 1 pip.
    
    Returns:
        list: Lista con el nivel del stop loss en cada momento.
    """
    stops = []
    mejor_precio = precios[0]
    distancia_precio = distancia_pips * 0.0001  # Asumiendo pares de 4 decimales
    
    for precio in precios:
        if direccion == 'long':
            if precio > mejor_precio:
                mejor_precio = precio
            stop_actual = mejor_precio - distancia_precio
        else: # short
            if precio < mejor_precio:
                mejor_precio = precio
            stop_actual = mejor_precio + distancia_precio
            
        stops.append(stop_actual)
        
    return stops


def trailing_stop_atr(df, multiplicador_atr=2.0, direccion='long'):
    """
    Calcula un Trailing Stop dinámico basado en el ATR (Average True Range).
    Es más robusto que el fijo porque se adapta a la volatilidad del mercado.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'close' y 'atr'.
        multiplicador_atr (float): Multiplicador del ATR para la distancia del stop.
        direccion (str): 'long' o 'short'.
    
    Returns:
        pd.Series: Serie con el nivel del stop loss en cada momento.
    """
    stops = []
    mejor_precio = df['close'].iloc[0]
    
    for i in range(len(df)):
        precio = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        distancia = atr * multiplicador_atr
        
        if direccion == 'long':
            if precio > mejor_precio:
                mejor_precio = precio
            stop_actual = mejor_precio - distancia
        else: # short
            if precio < mejor_precio:
                mejor_precio = precio
            stop_actual = mejor_precio + distancia
            
        stops.append(stop_actual)
        
    return pd.Series(stops, index=df.index)


def trailing_stop_media_movil(df, periodo_ma=20, direccion='long'):
    """
    Calcula un Trailing Stop basado en una Media Móvil.
    Muy usado en estrategias de seguimiento de tendencia.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'close'.
        periodo_ma (int): Período de la media móvil.
        direccion (str): 'long' o 'short'.
    
    Returns:
        pd.Series: Serie con el nivel del stop loss (la media móvil).
    """
    if direccion == 'long':
        # Para compras, el stop es la media móvil por debajo del precio
        return df['close'].rolling(periodo_ma).mean()
    else:
        # Para ventas, el stop es la media móvil por encima del precio
        return df['close'].rolling(periodo_ma).mean()


def simular_operacion_con_trailing(precios, stop_inicial, distancia_pips, direccion='long'):
    """
    Simula una operación completa aplicando un trailing stop fijo.
    
    Args:
        precios (list): Lista de precios.
        stop_inicial (float): Stop loss inicial de la operación.
        distancia_pips (float): Distancia en pips para el trailing.
        direccion (str): 'long' o 'short'.
    
    Returns:
        dict: Resultados de la simulación (precio_salida, beneficio, etc.).
    """
    stop_actual = stop_inicial
    distancia_precio = distancia_pips * 0.0001
    precio_entrada = precios[0]
    precio_salida = None
    operacion_abierta = True
    
    for i in range(1, len(precios)):
        precio = precios[i]
        
        # Actualizar trailing stop
        if direccion == 'long':
            stop_actual = max(stop_actual, precio - distancia_precio)
            # Comprobar si el stop ha sido tocado
            if precio <= stop_actual and operacion_abierta:
                precio_salida = stop_actual
                operacion_abierta = False
        else: # short
            stop_actual = min(stop_actual, precio + distancia_precio)
            if precio >= stop_actual and operacion_abierta:
                precio_salida = stop_actual
                operacion_abierta = False
                
    # Si al final del histórico la operación sigue abierta, cerrar al último precio
    if operacion_abierta:
        precio_salida = precios[-1]
        
    # Calcular beneficio en pips
    if direccion == 'long':
        beneficio_pips = (precio_salida - precio_entrada) / 0.0001
    else:
        beneficio_pips = (precio_entrada - precio_salida) / 0.0001
        
    return {
        'precio_entrada': precio_entrada,
        'precio_salida': precio_salida,
        'beneficio_pips': round(beneficio_pips, 1),
        'stop_final': stop_actual
    }


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo trailing_stops.py ===\n")
    
    # 1. Prueba de Trailing Stop Fijo
    print("1. Trailing Stop Fijo:")
    precios_ejemplo = [1.1000, 1.1010, 1.1020, 1.1015, 1.1030, 1.1025]
    stops = trailing_stop_fijo(precios_ejemplo, distancia_pips=10, direccion='long')
    print(f"Precios: {precios_ejemplo}")
    print(f"Stops:   {[round(s, 4) for s in stops]}\n")
    
    # 2. Prueba de Simulación de Operación
    print("2. Simulación de operación completa:")
    np.random.seed(42)
    # Generar una tendencia alcista con ruido
    tendencia = np.cumsum(np.random.randn(100) * 0.0005) + 0.0002 * np.arange(100)
    precios_sim = 1.1
