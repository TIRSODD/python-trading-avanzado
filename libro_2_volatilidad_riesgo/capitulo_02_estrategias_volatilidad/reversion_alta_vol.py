"""
reversion_alta_vol.py - Estrategia de reversión a la media en alta volatilidad

Esta estrategia opera en contra de movimientos extremos cuando el mercado 
está en régimen de alta volatilidad, aprovechando que los precios tienden 
a volver a la media.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 2: Estrategias Basadas en Volatilidad
"""

import numpy as np
import pandas as pd


def estrategia_reversion_alta_volatilidad(df, periodo_ma=50, umbral_volatilidad=0.75, 
                                          multiplicador_std=2.0, capital=10000, riesgo_pct=1.0):
    """
    Estrategia de reversión a la media en régimen de alta volatilidad.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC y 'atr_pct'.
        periodo_ma (int): Período de la media móvil central.
        umbral_volatilidad (float): Percentil para definir alta volatilidad.
        multiplicador_std (float): Multiplicador para las Bandas de Bollinger.
        capital (float): Capital disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación.
    
    Returns:
        pd.DataFrame: DataFrame con las señales generadas.
    """
    # Calcular media móvil y desviación estándar
    df['ma'] = df['close'].rolling(periodo_ma).mean()
    df['std'] = df['close'].rolling(periodo_ma).std()
    
    # Calcular bandas (Bollinger)
    df['banda_superior'] = df['ma'] + (multiplicador_std * df['std'])
    df['banda_inferior'] = df['ma'] - (multiplicador_std * df['std'])
    
    # Umbrales de volatilidad
    umbral_alto = df['atr_pct'].quantile(umbral_volatilidad)
    
    señales = []
    
    for i in range(periodo_ma, len(df)):
        # Solo operar en alta volatilidad
        if df['atr_pct'].iloc[i] < umbral_alto:
            continue
            
        precio = df['close'].iloc[i]
        precio_anterior = df['close'].iloc[i-1]
        ma = df['ma'].iloc[i]
        banda_sup = df['banda_superior'].iloc[i]
        banda_inf = df['banda_inferior'].iloc[i]
        
        # Señal de venta (sobrecompra) - El precio cruza la banda superior hacia abajo
        if precio_anterior > banda_sup and precio <= banda_sup:
            stop_loss = banda_sup + (df['atr'].iloc[i] * 1.5)
            take_profit = ma
            
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
                
        # Señal de compra (sobreventa) - El precio cruza la banda inferior hacia arriba
        elif precio_anterior < banda_inf and precio >= banda_inf:
            stop_loss = banda_inf - (df['atr'].iloc[i] * 1.5)
            take_profit = ma
            
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
                
    df_señales = pd.DataFrame(señales)
    
    if len(df_señales) > 0:
        print(f"=== ESTRATEGIA REVERSIÓN ALTA VOLATILIDAD ===")
        print(f"Total señales: {len(df_señales)}")
        print(f"Señales long: {len(df_señales[df_señales['direccion'] == 'long'])}")
        print(f"Señales short: {len(df_señales[df_señales['direccion'] == 'short'])}")
    
    return df_señales


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo reversion_alta_vol.py ===\n")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n = 500
    
    df_ejemplo = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n)),
        'high': 100 + np.cumsum(np.random.randn(n)) + np.random.uniform(0, 2, n),
        'low': 100 + np.cumsum(np.random.randn(n)) - np.random.uniform(0, 2, n),
        'close': 100 + np.cumsum(np.random.randn(n))
    })
    
    # Simular ATR y ATR%
    df_ejemplo['atr'] = np.abs(np.random.randn(n)) * 2 + 1
    df_ejemplo['atr_pct'] = (df_ejemplo['atr'] / df_ejemplo['close']) * 100
    
    # Ejecutar estrategia
    señales = estrategia_reversion_alta_volatilidad(df_ejemplo)
    
    print("\n✓ Prueba completada exitosamente")
