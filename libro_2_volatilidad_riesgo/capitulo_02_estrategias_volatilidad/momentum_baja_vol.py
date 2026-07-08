"""
momentum_baja_vol.py - Estrategia de momentum en baja volatilidad

Esta estrategia opera rupturas de rangos de consolidación cuando el mercado 
está en régimen de baja volatilidad, aprovechando que los movimientos 
direccionales tienden a continuar.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 2: Estrategias Basadas en Volatilidad
"""

import numpy as np
import pandas as pd


def estrategia_momentum_baja_volatilidad(df, periodo_ma=50, umbral_volatilidad=0.25,
                                         periodo_ruptura=20, capital=10000, riesgo_pct=1.0):
    """
    Estrategia de momentum en régimen de baja volatilidad.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC y 'atr_pct'.
        periodo_ma (int): Período de la media móvil de tendencia.
        umbral_volatilidad (float): Percentil para definir baja volatilidad.
        periodo_ruptura (int): Período para calcular el rango de consolidación.
        capital (float): Capital disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación.
    
    Returns:
        pd.DataFrame: DataFrame con las señales generadas.
    """
    # Calcular media móvil
    df['ma'] = df['close'].rolling(periodo_ma).mean()
    
    # Calcular rango de consolidación
    df['max_consolidacion'] = df['high'].rolling(periodo_ruptura).max()
    df['min_consolidacion'] = df['low'].rolling(periodo_ruptura).min()
    
    # Umbrales de volatilidad
    umbral_bajo = df['atr_pct'].quantile(umbral_volatilidad)
    
    señales = []
    
    for i in range(max(periodo_ma, periodo_ruptura), len(df)):
        # Solo operar en baja volatilidad
        if df['atr_pct'].iloc[i] > umbral_bajo:
            continue
            
        precio = df['close'].iloc[i]
        max_cons = df['max_consolidacion'].iloc[i-1]
        min_cons = df['min_consolidacion'].iloc[i-1]
        ma = df['ma'].iloc[i]
        
        # Ruptura alcista del rango de consolidación
        if precio > max_cons:
            # Confirmación: precio por encima de la media móvil
            if precio > ma:
                stop_loss = min_cons  # Stop en el mínimo del rango
                take_profit = precio + (precio - min_cons) * 2  # Ratio 2:1
                
                riesgo_pips = abs(stop_loss - precio) * 10000
                riesgo_monetario = capital * (riesgo_pct / 100)
                tamaño_lotes = riesgo_monetario / (riesgo_pips * 10) if riesgo_pips > 0 else 0
                
                señales.append({
                    'timestamp': df.index[i],
                    'direccion': 'long',
                    'precio_entrada': precio,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': df['atr'].iloc[i],
                    'tamaño_lotes': round(tamaño_lotes, 2)
                })
                
        # Ruptura bajista del rango de consolidación
        elif precio < min_cons:
            # Confirmación: precio por debajo de la media móvil
            if precio < ma:
                stop_loss = max_cons  # Stop en el máximo del rango
                take_profit = precio - (max_cons - precio) * 2  # Ratio 2:1
                
                riesgo_pips = abs(stop_loss - precio) * 10000
                riesgo_monetario = capital * (riesgo_pct / 100)
                tamaño_lotes = riesgo_monetario / (riesgo_pips * 10) if riesgo_pips > 0 else 0
                
                señales.append({
                    'timestamp': df.index[i],
                    'direccion': 'short',
                    'precio_entrada': precio,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': df['atr'].iloc[i],
                    'tamaño_lotes': round(tamaño_lotes, 2)
                })
                
    df_señales = pd.DataFrame(señales)
    
    if len(df_señales) > 0:
        print(f"=== ESTRATEGIA MOMENTUM BAJA VOLATILIDAD ===")
        print(f"Total señales: {len(df_señales)}")
        print(f"Señales long: {len(df_señales[df_señales['direccion'] == 'long'])}")
        print(f"Señales short: {len(df_señales[df_señales['direccion'] == 'short'])}")
    
    return df_señales


def detectar_rango_consolidacion(df, periodo=20, umbral_amplitud=0.02):
    """
    Detecta períodos de consolidación basados en la amplitud del rango.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC.
        periodo (int): Ventana de tiempo para calcular el rango.
        umbral_amplitud (float): Umbral de amplitud para considerar consolidación.
    
    Returns:
        pd.DataFrame: DataFrame con columna 'en_consolidacion' añadida.
    """
    # Calcular rango máximo y mínimo
    df['max_rango'] = df['high'].rolling(periodo).max()
    df['min_rango'] = df['low'].rolling(periodo).min()
    
    # Calcular amplitud del rango como porcentaje
    df['amplitud_rango'] = (df['max_rango'] - df['min_rango']) / df['min_rango']
    
    # Marcar períodos de consolidación
    df['en_consolidacion'] = df['amplitud_rango'] < umbral_amplitud
    
    return df


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo momentum_baja_vol.py ===\n")
    
    # Generar datos sintéticos con período de consolidación
    np.random.seed(42)
    n = 500
    
    # Crear un período de consolidación en el medio
    precio_base = 100
    precios = [precio_base]
    
    for i in range(1, n):
        if 200 <= i <= 300:
            # Consolidación: movimientos pequeños alrededor de 100
            cambio = np.random.uniform(-0.5, 0.5)
        else:
            # Tendencia normal
            cambio = np.random.randn() * 2
            
        precios.append(precios[-1] + cambio)
    
    df_ejemplo = pd.DataFrame({
        'open': precios,
        'high': [p + abs(np.random.uniform(0, 2)) for p in precios],
        'low': [p - abs(np.random.uniform(0, 2)) for p in precios],
        'close': precios
    })
    
    # Simular ATR y ATR%
    df_ejemplo['atr'] = np.abs(np.random.randn(n)) * 2 + 1
    df_ejemplo['atr_pct'] = (df_ejemplo['atr'] / df_ejemplo['close']) * 100
    
    # Ejecutar estrategia
    señales = estrategia_momentum_baja_volatilidad(df_ejemplo)
    
    print("\n✓ Prueba completada exitosamente")
