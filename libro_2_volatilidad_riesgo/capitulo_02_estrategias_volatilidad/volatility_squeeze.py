"""
volatility_squeeze.py - Estrategia Volatility Squeeze (Compresión de Volatilidad)

Esta estrategia detecta períodos donde las Bandas de Bollinger se estrechan 
dentro de los Canales de Keltner, señalando que la volatilidad está comprimida 
y se avecina un movimiento explosivo.

Basada en el concepto original de John Carter (TTM Squeeze).

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 2: Estrategias Basadas en Volatilidad
"""

import numpy as np
import pandas as pd


def calcular_canales_keltner(df, periodo=20, multiplicador_atr=1.5):
    """
    Calcula los Canales de Keltner (EMA ± ATR * multiplicador).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC.
        periodo (int): Período de la EMA.
        multiplicador_atr (float): Multiplicador del ATR.
    
    Returns:
        pd.DataFrame: DataFrame con columnas de Keltner añadidas.
    """
    # EMA del precio de cierre
    df['ema'] = df['close'].ewm(span=periodo, adjust=False).mean()
    
    # Calcular ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = df['tr'].rolling(periodo).mean()
    
    # Canales de Keltner
    df['keltner_superior'] = df['ema'] + (multiplicador_atr * df['atr'])
    df['keltner_inferior'] = df['ema'] - (multiplicador_atr * df['atr'])
    
    return df


def detectar_squeeze(df, periodo_bollinger=20, multiplicador_bollinger=2.0,
                     periodo_keltner=20, multiplicador_keltner=1.5):
    """
    Detecta períodos de Squeeze (Bollinger dentro de Keltner).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC.
        periodo_bollinger (int): Período de Bandas de Bollinger.
        multiplicador_bollinger (float): Multiplicador de Bollinger.
        periodo_keltner (int): Período de Canales de Keltner.
        multiplicador_keltner (float): Multiplicador de Keltner.
    
    Returns:
        pd.DataFrame: DataFrame con columna 'squeeze' añadida.
    """
    # Calcular Bandas de Bollinger
    df['bb_ma'] = df['close'].rolling(periodo_bollinger).mean()
    df['bb_std'] = df['close'].rolling(periodo_bollinger).std()
    df['bb_superior'] = df['bb_ma'] + (multiplicador_bollinger * df['bb_std'])
    df['bb_inferior'] = df['bb_ma'] - (multiplicador_bollinger * df['bb_std'])
    
    # Calcular Canales de Keltner
    df = calcular_canales_keltner(df, periodo_keltner, multiplicador_keltner)
    
    # Detectar Squeeze: Bollinger dentro de Keltner
    df['squeeze'] = (df['bb_inferior'] > df['keltner_inferior']) & \
                    (df['bb_superior'] < df['keltner_superior'])
    
    # Detectar liberación del Squeeze (cuando Bollinger sale de Keltner)
    df['squeeze_liberado'] = (df['bb_inferior'] < df['keltner_inferior']) | \
                             (df['bb_superior'] > df['keltner_superior'])
    
    return df


def estrategia_volatility_squeeze(df, periodo_bollinger=20, multiplicador_bollinger=2.0,
                                   periodo_keltner=20, multiplicador_keltner=1.5,
                                   capital=10000, riesgo_pct=1.0):
    """
    Estrategia completa Volatility Squeeze.
    
    Opera cuando el squeeze se libera, en la dirección del momentum.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC.
        periodo_bollinger (int): Período de Bandas de Bollinger.
        multiplicador_bollinger (float): Multiplicador de Bollinger.
        periodo_keltner (int): Período de Canales de Keltner.
        multiplicador_keltner (float): Multiplicador de Keltner.
        capital (float): Capital disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación.
    
    Returns:
        pd.DataFrame: DataFrame con las señales generadas.
    """
    # Detectar squeeze
    df = detectar_squeeze(df, periodo_bollinger, multiplicador_bollinger,
                          periodo_keltner, multiplicador_keltner)
    
    # Calcular momentum (Linear Regression Slope simplificado)
    df['momentum'] = df['close'] - df['close'].rolling(periodo_bollinger).mean()
    
    señales = []
    en_squeeze = False
    
    for i in range(periodo_bollinger, len(df)):
        # Detectar inicio de squeeze
        if df['squeeze'].iloc[i] and not en_squeeze:
            en_squeeze = True
            continue
            
        # Detectar liberación del squeeze
        if df['squeeze_liberado'].iloc[i] and en_squeeze:
            en_squeeze = False
            
            precio = df['close'].iloc[i]
            momentum = df['momentum'].iloc[i]
            atr = df['atr'].iloc[i]
            
            # Dirección basada en momentum
            if momentum > 0:
                direccion = 'long'
                stop_loss = precio - (atr * 2)
                take_profit = precio + (atr * 4)  # Ratio 2:1
            else:
                direccion = 'short'
                stop_loss = precio + (atr * 2)
                take_profit = precio - (atr * 4)  # Ratio 2:1
            
            # Calcular tamaño de posición
            riesgo_pips = abs(stop_loss - precio) * 10000
            riesgo_monetario = capital * (riesgo_pct / 100)
            tamaño_lotes = riesgo_monetario / (riesgo_pips * 10) if riesgo_pips > 0 else 0
            
            señales.append({
                'timestamp': df.index[i],
                'direccion': direccion,
                'precio_entrada': precio,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr,
                'tamaño_lotes': round(tamaño_lotes, 2),
                'momentum': momentum
            })
            
    df_señales = pd.DataFrame(señales)
    
    if len(df_señales) > 0:
        print(f"=== ESTRATEGIA VOLATILITY SQUEEZE ===")
        print(f"Total señales: {len(df_señales)}")
        print(f"Señales long: {len(df_señales[df_señales['direccion'] == 'long'])}")
        print(f"Señales short: {len(df_señales[df_señales['direccion'] == 'short'])}")
        
        # Estadísticas del momentum
        momentum_medio = df_señales['momentum'].mean()
        print(f"Momentum medio en señales: {momentum_medio:.4f}")
    
    return df_señales


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo volatility_squeeze.py ===\n")
    
    # Generar datos sintéticos con período de squeeze
    np.random.seed(42)
    n = 500
    
    precios = [100]
    for i in range(1, n):
        if 200 <= i <= 250:
            # Período de squeeze: movimientos muy pequeños
            cambio = np.random.uniform(-0.1, 0.1)
        elif i > 250:
            # Explosión de volatilidad
            cambio = np.random.randn() * 3
        else:
            cambio = np.random.randn() * 1.5
            
        precios.append(precios[-1] + cambio)
    
    df_ejemplo = pd.DataFrame({
        'open': precios,
        'high': [p + abs(np.random.uniform(0, 1.5)) for p in precios],
        'low': [p - abs(np.random.uniform(0, 1.5)) for p in precios],
        'close': precios
    })
    
    # Ejecutar estrategia
    señales = estrategia_volatility_squeeze(df_ejemplo)
    
    print("\n✓ Prueba completada exitosamente")
