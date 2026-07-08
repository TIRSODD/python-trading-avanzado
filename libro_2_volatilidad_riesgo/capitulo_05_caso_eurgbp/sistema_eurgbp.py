"""
sistema_eurgbp.py - Sistema de Trading para el par EUR/GBP

Este módulo implementa un sistema completo adaptado a las características 
del par EUR/GBP. Este par suele tener un comportamiento de reversión a la 
media muy marcado, especialmente durante la sesión europea.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 5: Caso Práctico - EUR/GBP
"""

import numpy as np
import pandas as pd


class SistemaEURGBP:
    """
    Clase principal del sistema de trading para EUR/GBP.
    Combina filtros de volatilidad, horarios y reversión a la media.
    """
    
    def __init__(self, capital_inicial=10000, riesgo_pct=1.0):
        self.capital = capital_inicial
        self.riesgo_pct = riesgo_pct
        self.posicion_abierta = False
        self.direccion = None
        self.precio_entrada = 0
        self.stop_loss = 0
        self.take_profit = 0
        
    def calcular_indicadores(self, df):
        """
        Calcula todos los indicadores necesarios para el sistema.
        """
        # Medias móviles para la tendencia de fondo
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Bandas de Bollinger para la reversión a la media
        df['bb_media'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_superior'] = df['bb_media'] + (2.0 * df['bb_std'])
        df['bb_inferior'] = df['bb_media'] - (2.0 * df['bb_std'])
        
        # ATR para stops dinámicos
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # RSI para confirmar sobrecompra/sobreventa
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def evaluar_entrada(self, row, row_prev):
        """
        Evalúa si se dan las condiciones para abrir una operación.
        """
        # Filtro 1: No operar si hay tendencia fuerte (EMA50 y EMA200 muy separadas)
        if abs(row['ema_50'] - row['ema_200']) > (row['atr'] * 3):
            return None
            
        # Filtro 2: RSI en extremos
        if row['rsi'] > 70 and row_prev['rsi'] <= 70:
            return 'short' # Sobrecompra
        elif row['rsi'] < 30 and row_prev['rsi'] >= 30:
            return 'long' # Sobreventa
            
        # Filtro 3: Toque de Bandas de Bollinger
        if row['close'] > row['bb_superior'] and row_prev['close'] <= row_prev['bb_superior']:
            if row['rsi'] > 60: # Confirmación
                return 'short'
        elif row['close'] < row['bb_inferior'] and row_prev['close'] >= row_prev['bb_inferior']:
            if row['rsi'] < 40: # Confirmación
                return 'long'
                
        return None

    def gestionar_riesgo(self, precio_entrada, direccion, atr_actual):
        """
        Calcula el tamaño de la posición y los niveles de Stop/Take Profit.
        """
        if direccion == 'long':
            self.stop_loss = precio_entrada - (atr_actual * 2.0)
            self.take_profit = precio_entrada + (atr_actual * 3.0) # Ratio 1.5:1
        else:
            self.stop_loss = precio_entrada + (atr_actual * 2.0)
            self.take_profit = precio_entrada - (atr_actual * 3.0)
            
        # Calcular tamaño (asumiendo valor_pip = 10 para lotes estándar)
        riesgo_monetario = self.capital * (self.riesgo_pct / 100)
        stop_pips = abs(self.stop_loss - precio_entrada) * 10000
        lotes = riesgo_monetario / (stop_pips * 10) if stop_pips > 0 else 0
        
        return round(lotes, 2)

    def simular_sistema(self, df):
        """
        Ejecuta el sistema sobre un DataFrame histórico.
        """
        df = self.calcular_indicadores(df)
        operaciones = []
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            row_prev = df.iloc[i-1]
            
            if not self.posicion_abierta:
                señal = self.evaluar_entrada(row, row_prev)
                if señal:
                    self.posicion_abierta = True
                    self.direccion = señal
                    self.precio_entrada = row['close']
                    self.lotes = self.gestionar_riesgo(row['close'], señal, row['atr'])
                    
                    operaciones.append({
                        'timestamp': df.index[i],
                        'tipo': 'ENTRADA',
                        'direccion': señal,
                        'precio': row['close'],
                        'lotes': self.lotes
                    })
            else:
                # Comprobar salidas
                salir = False
                motivo = ""
                
                if self.direccion == 'long':
                    if row['low'] <= self.stop_loss:
                        salir = True
                        precio_salida = self.stop_loss
                        motivo = "STOP LOSS"
                    elif row['high'] >= self.take_profit:
                        salir = True
                        precio_salida = self.take_profit
                        motivo = "TAKE PROFIT"
                else: # short
                    if row['high'] >= self.stop_loss:
                        salir = True
                        precio_salida = self.stop_loss
                        motivo = "STOP LOSS"
                    elif row['low'] <= self.take_profit:
                        salir = True
                        precio_salida = self.take_profit
                        motivo = "TAKE PROFIT"
                        
                if salir:
                    self.posicion_abierta = False
                    pnl = (precio_salida - self.precio_entrada) * 10000 * self.lotes * 10
                    if self.direccion == 'short':
                        pnl = -pnl
                        
                    operaciones.append({
                        'timestamp': df.index[i],
                        'tipo': 'SALIDA',
                        'motivo': motivo,
                        'precio': precio_salida,
                        'pnl': pnl
                    })
                    
        return pd.DataFrame(operaciones)


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando Sistema EUR/GBP ===\n")
    
    # Generar datos sintéticos que simulen EUR/GBP (rango lateral con ruido)
    np.random.seed(42)
    n = 1000
    precios = [0.8500]
    for i in range(1, n):
        # Sesgo muy leve para simular el comportamiento real
        cambio = np.random.normal(0, 0.0005) 
        precios.append(precios[-1] + cambio)
        
    df_test = pd.DataFrame({
        'open': precios,
        'high': [p + abs(np.random.normal(0, 0.0003)) for p in precios],
        'low': [p - abs(np.random.normal(0, 0.0003)) for p in precios],
        'close': precios
    })
    
    sistema = SistemaEURGBP(capital_inicial=10000, riesgo_pct=1.0)
    resultados = sistema.simular_sistema(df_test)
    
    if len(resultados) > 0:
        print(f"Total eventos (entradas + salidas): {len(resultados)}")
        entradas = resultados[resultados['tipo'] == 'ENTRADA']
        print(f"Operaciones abiertas: {len(entradas)}")
        
        salidas = resultados[resultados['tipo'] == 'SALIDA']
        if len(salidas) > 0:
            pnl_total = salidas['pnl'].sum()
            print(f"PnL Total simulado: {pnl_total:.2f}€")
    else:
        print("No se generaron operaciones en este periodo sintético.")
        
    print("\n✓ Prueba completada exitosamente")
