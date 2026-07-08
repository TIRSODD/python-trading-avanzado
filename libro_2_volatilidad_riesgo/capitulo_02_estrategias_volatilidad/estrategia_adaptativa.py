"""
estrategia_adaptativa.py - Estrategia Maestra Adaptativa

Este módulo actúa como un "orquestador". Analiza el régimen de volatilidad 
actual del mercado y decide automáticamente qué sub-estrategia activar:
- Alta volatilidad -> Reversión a la media
- Baja volatilidad -> Momentum / Rupturas
- Volatilidad normal -> Esperar / Operar con parámetros estándar

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 2: Estrategias Basadas en Volatilidad
"""

import numpy as np
import pandas as pd


def calcular_regimen_actual(df, periodo=14):
    """
    Calcula el régimen de volatilidad actual basado en percentiles históricos.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'atr_pct'.
        periodo (int): Período para calcular el ATR si no existe.
    
    Returns:
        str: 'alta', 'baja' o 'normal'.
    """
    if 'atr_pct' not in df.columns:
        # Cálculo rápido de ATR% si no existe
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(periodo).mean()
        df['atr_pct'] = (atr / df['close']) * 100
        
    atr_pct = df['atr_pct'].dropna()
    
    # Umbrales dinámicos
    umbral_bajo = atr_pct.quantile(0.25)
    umbral_alto = atr_pct.quantile(0.75)
    
    valor_actual = df['atr_pct'].iloc[-1]
    
    if valor_actual > umbral_alto:
        return 'alta'
    elif valor_actual < umbral_bajo:
        return 'baja'
    else:
        return 'normal'


def estrategia_adaptativa(df, capital=10000, riesgo_pct=1.0):
    """
    Función principal que adapta la lógica de trading al régimen de mercado.
    
    En un entorno de producción real, esta función importaría y llamaría a:
    - reversion_alta_volatilidad()
    - momentum_baja_volatilidad()
    
    Aquí simulamos la lógica de decisión para mantener el código autocontenido.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas OHLC.
        capital (float): Capital disponible.
        riesgo_pct (float): Porcentaje de riesgo por operación.
    
    Returns:
        dict: Diccionario con la decisión tomada, régimen detectado y parámetros.
    """
    # 1. Detectar régimen actual
    regimen = calcular_regimen_actual(df)
    
    # 2. Tomar decisión basada en el régimen
    decision = {
        'timestamp': df.index[-1],
        'regimen': regimen,
        'estrategia_activa': None,
        'parametros': {},
        'accion': 'ESPERAR'
    }
    
    if regimen == 'alta':
        decision['estrategia_activa'] = 'Reversión a la Media'
        decision['accion'] = 'BUSCAR_SOBRECOMPRA_SOBREVENTA'
        # Parámetros específicos para alta volatilidad (stops más amplios)
        decision['parametros'] = {
            'multiplicador_stop': 2.5,
            'multiplicador_tp': 2.0,
            'tamaño_posicion_reducido': True
        }
        
    elif regimen == 'baja':
        decision['estrategia_activa'] = 'Momentum / Ruptura'
        decision['accion'] = 'BUSCAR_RUPTURAS_RANGO'
        # Parámetros específicos para baja volatilidad (stops más ajustados)
        decision['parametros'] = {
            'multiplicador_stop': 1.5,
            'multiplicador_tp': 3.0,
            'tamaño_posicion_reducido': False
        }
        
    else: # normal
        decision['estrategia_activa'] = 'Híbrida / Conservadora'
        decision['accion'] = 'OPERAR_SOLO_TENDENCIA_CLARA'
        decision['parametros'] = {
            'multiplicador_stop': 2.0,
            'multiplicador_tp': 2.0,
            'tamaño_posicion_reducido': False
        }
        
    # 3. Mostrar resultado en consola
    print(f"=== ANÁLISIS ADAPTATIVO ===")
    print(f"Fecha: {decision['timestamp']}")
    print(f"Régimen detectado: {regimen.upper()}")
    print(f"Estrategia activada: {decision['estrategia_activa']}")
    print(f"Acción: {decision['accion']}")
    print(f"Parámetros: {decision['parametros']}\n")
    
    return decision


def simular_historico_adaptativo(df, capital_inicial=10000):
    """
    Simula cómo se habría comportado la estrategia adaptativa en el pasado.
    
    Args:
        df (pd.DataFrame): DataFrame histórico con OHLC.
        capital_inicial (float): Capital inicial para la simulación.
    
    Returns:
        pd.DataFrame: DataFrame con la equity curve y decisiones diarias.
    """
    decisiones = []
    capital = capital_inicial
    
    # Simulación simplificada (en un backtest real usaríamos el motor de backtesting)
    for i in range(50, len(df)):
        df_window = df.iloc[:i+1]
        decision = estrategia_adaptativa(df_window)
        decision['capital'] = capital
        
        # Simulación muy básica de PnL aleatorio para el ejemplo
        pnl = np.random.normal(10, 50) 
        capital += pnl
        decision['pnl_simulado'] = pnl
        
        decisiones.append(decision)
        
    return pd.DataFrame(decisiones)


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo estrategia_adaptativa.py ===\n")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n = 200
    
    df_ejemplo = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n)),
        'high': 100 + np.cumsum(np.random.randn(n)) + np.random.uniform(0, 2, n),
        'low': 100 + np.cumsum(np.random.randn(n)) - np.random.uniform(0, 2, n),
        'close': 100 + np.cumsum(np.random.randn(n))
    })
    
    # Probar análisis puntual
    print("1. Análisis puntual del último día:")
    decision = estrategia_adaptativa(df_ejemplo)
    
    # Probar simulación histórica
    print("\n2. Simulación histórica (primeras 5 decisiones):")
    df_sim = simular_historico_adaptativo(df_ejemplo)
    print(df_sim[['timestamp', 'regimen', 'estrategia_activa', 'pnl_simulado']].head())
    
    print("\n✓ Prueba completada exitosamente")
