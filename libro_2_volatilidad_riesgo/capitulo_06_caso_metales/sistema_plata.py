"""
sistema_plata.py - Sistema de Trading para Plata (XAG/USD)

La Plata es más volátil y "ruidosa" que el Oro. Este sistema utiliza 
un enfoque de seguimiento de tendencia (Trend Following) con filtros 
de volatilidad para evitar entrar en mercados laterales, y utiliza 
stops más amplios para soportar el ruido característico de la plata.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 6: Caso Práctico - Metales
"""

import numpy as np
import pandas as pd


class SistemaPlata:
    """
    Clase principal del sistema de trading para Plata (XAG/USD).
    """
    
    def __init__(self, capital_inicial=10000, riesgo_pct=1.0):
        self.capital = capital_inicial
        self.riesgo_pct = riesgo_pct
        self.posicion_abierta = False
        self.direccion = None
        self.precio_entrada = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.lotes = 0
        
    def calcular_indicadores(self, df):
        """
        Calcula los indicadores específicos para Plata.
        Usamos ADX para filtrar mercados laterales (la plata mata las cuentas en rangos).
        """
        # Medias móviles para tendencia
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # ATR para volatilidad
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Cálculo simplificado de ADX para medir fuerza de tendencia
        # (En un entorno real usaríamos la librería 'ta' o 'pandas-ta')
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        
        df['atr_smooth'] = df['tr'].rolling(14).mean()
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / df['atr_smooth'])
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / df['atr_smooth'])
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        
        return df

    def evaluar_entrada(self, row, row_prev):
        """
        Evalúa condiciones de entrada.
        Solo operamos si el ADX indica que hay una tendencia fuerte (ADX > 25).
        """
        # Filtro de fuerza de tendencia
        if row['adx'] < 25:
            return None
            
        # Cruce de EMAs (La EMA 20 cruza la EMA 50)
        cruce_alcista = (row['ema_20'] > row['ema_50']) and (row_prev['ema_20'] <= row_prev['ema_50'])
        cruce_bajista = (row['ema_20'] < row['ema_50']) and (row_prev['ema_20'] >= row_prev['ema_50'])
        
        if cruce_alcista:
            return 'long'
        elif cruce_bajista:
            return 'short'
            
        return None

    def gestionar_riesgo(self, precio_entrada, direccion, atr_actual):
        """
        Calcula el tamaño de la posición.
        La plata requiere stops más amplios (3.0 ATR) debido a su ruido.
        """
        if direccion == 'long':
            self.stop_loss = precio_entrada - (atr_actual * 3.0)
            self.take_profit = precio_entrada + (atr_actual * 6.0) # Ratio 1:2 para compensar menor win-rate
        else:
            self.stop_loss = precio_entrada + (atr_actual * 3.0)
            self.take_profit = precio_entrada - (atr_actual * 6.0)
            
        # Calcular tamaño (Asumiendo 1 lote Plata = 5000 onzas, 1$ movimiento = 5000$)
        # Ajustamos a mini-lotes para cuentas retail: 1 lote = 100 onzas (1$ = 100$)
        riesgo_monetario = self.capital * (self.riesgo_pct / 100)
        stop_dolares = abs(self.stop_loss - precio_entrada)
        lotes = riesgo_monetario / (stop_dolares * 100) if stop_dolares > 0 else 0
        
        self.lotes = round(lotes, 2)
        return self.lotes

    def simular_sistema(self, df):
        """
        Ejecuta el sistema sobre un DataFrame histórico.
        """
        df = self.calcular_indicadores(df)
        operaciones = []
        
        # Empezamos en 100 para dar tiempo a que se calculen los indicadores
        for i in range(100, len(df)):
            row = df.iloc[i]
            row_prev = df.iloc[i-1]
            
            if not self.posicion_abierta:
                señal = self.evaluar_entrada(row, row_prev)
                if señal:
                    self.posicion_abierta = True
                    self.direccion = señal
                    self.precio_entrada = row['close']
                    self.gestionar_riesgo(row['close'], señal, row['atr'])
                    
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
                precio_salida = 0
                
                if self.direccion == 'long':
                    if row['low'] <= self.stop_loss:
                        salir = True; precio_salida = self.stop_loss; motivo = "STOP LOSS"
                    elif row['high'] >= self.take_profit:
                        salir = True; precio_salida = self.take_profit; motivo = "TAKE PROFIT"
                else: # short
                    if row['high'] >= self.stop_loss:
                        salir = True; precio_salida = self.stop_loss; motivo = "STOP LOSS"
                    elif row['low'] <= self.take_profit:
                        salir = True; precio_salida = self.take_profit; motivo = "TAKE PROFIT"
                        
                if salir:
                    self.posicion_abierta = False
                    pnl = (precio_salida - self.precio_entrada) * 100 * self.lotes
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
    print("=== Probando Sistema Plata (XAG/USD) ===\n")
    
    # Generar datos sintéticos que simulen la alta volatilidad de la plata
    np.random.seed(42)
    n = 1000
    # Tendencia con fuertes sacudidas (ruido)
    tendencia = np.linspace(0, 10, n) 
    ruido_fuerte = np.cumsum(np.random.randn(n) * 0.5) 
    precios = 25.0 + tendencia + ruido_fuerte # Precio base de la Plata ~25$
        
    df_test = pd.DataFrame({
        'open': precios,
        'high': precios + abs(np.random.normal(0, 0.3, n)),
        'low': precios - abs(np.random.normal(0, 0.3, n)),
        'close': precios
    })
    
    sistema = SistemaPlata(capital_inicial=10000, riesgo_pct=1.0)
    resultados = sistema.simular_sistema(df_test)
    
    if len(resultados) > 0:
        print(f"Total eventos (entradas + salidas): {len(resultados)}")
        entradas = resultados[resultados['tipo'] == 'ENTRADA']
        print(f"Operaciones abiertas: {len(entradas)}")
        
        salidas = resultados[resultados['tipo'] == 'SALIDA']
        if len(salidas) > 0:
            pnl_total = salidas['pnl'].sum()
            print(f"PnL Total simulado: ${pnl_total:.2f}")
    else:
        print("No se generaron operaciones en este periodo sintético.")
        
    print("\n✓ Prueba completada exitosamente")
