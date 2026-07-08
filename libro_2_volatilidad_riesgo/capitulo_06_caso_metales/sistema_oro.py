"""
sistema_oro.py - Sistema de Trading para Oro (XAU/USD)

El Oro (Gold) es un activo con una fuerte tendencia y alta volatilidad.
Este sistema está diseñado para operar rupturas y retrocesos (pullbacks) 
en la dirección de la tendencia principal, usando medias móviles y ATR.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 6: Caso Práctico - Metales
"""

import numpy as np
import pandas as pd


class SistemaOro:
    """
    Clase principal del sistema de trading para Oro (XAU/USD).
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
        Calcula los indicadores específicos para Oro.
        El Oro respeta mucho la EMA de 50 y 200 periodos.
        """
        # Medias móviles exponenciales para tendencia
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ATR para volatilidad (El oro es muy volátil, usamos periodo 14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Volumen (simulado o real) para confirmar rupturas
        # En este ejemplo usamos el rango de la vela como proxy de volumen/volatilidad intradía
        df['rango_vela'] = df['high'] - df['low']
        
        return df

    def evaluar_entrada(self, row, row_prev):
        """
        Evalúa condiciones de entrada (Pullback a la EMA 50 en tendencia).
        """
        # Filtro de Tendencia Principal
        tendencia_alcista = row['ema_50'] > row['ema_200']
        tendencia_bajista = row['ema_50'] < row['ema_200']
        
        # Estrategia: Comprar en retroceso a la EMA 50 en tendencia alcista
        if tendencia_alcista:
            # Si el precio toca o se acerca mucho a la EMA 50 desde arriba
            if row['low'] <= row['ema_50'] * 1.002 and row['close'] > row['ema_50']:
                # Confirmación: la vela anterior era bajista y la actual es alcista
                if row_prev['close'] < row_prev['open'] and row['close'] > row['open']:
                    return 'long'
                    
        # Estrategia: Vender en retroceso a la EMA 50 en tendencia bajista
        elif tendencia_bajista:
            if row['high'] >= row['ema_50'] * 0.998 and row['close'] < row['ema_50']:
                if row_prev['close'] > row_prev['open'] and row['close'] < row['open']:
                    return 'short'
                    
        return None

    def gestionar_riesgo(self, precio_entrada, direccion, atr_actual):
        """
        Calcula el tamaño de la posición. El Oro requiere stops más amplios.
        """
        # Usamos un multiplicador de 2.5 ATR para el stop (Oro es volátil)
        if direccion == 'long':
            self.stop_loss = precio_entrada - (atr_actual * 2.5)
            self.take_profit = precio_entrada + (atr_actual * 4.0) # Ratio 1.6:1
        else:
            self.stop_loss = precio_entrada + (atr_actual * 2.5)
            self.take_profit = precio_entrada - (atr_actual * 4.0)
            
        # Calcular tamaño (Asumiendo 1 lote Oro = 100 onzas, 1$ movimiento = 100$)
        riesgo_monetario = self.capital * (self.riesgo_pct / 100)
        stop_dolares = abs(self.stop_loss - precio_entrada)
        # Fórmula simplificada para Oro: lotes = riesgo / (stop_dolares * 100)
        lotes = riesgo_monetario / (stop_dolares * 100) if stop_dolares > 0 else 0
        
        return round(lotes, 2)

    def simular_sistema(self, df):
        """
        Ejecuta el sistema sobre un DataFrame histórico.
        """
        df = self.calcular_indicadores(df)
        operaciones = []
        
        # Empezamos en 200 para dar tiempo a que se calculen las EMAs
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
                    # PnL para Oro: (precio_salida - precio_entrada) * 100 * lotes
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
    print("=== Probando Sistema Oro (XAU/USD) ===\n")
    
    # Generar datos sintéticos que simulen una tendencia alcista con retrocesos (típico del Oro)
    np.random.seed(42)
    n = 1000
    tendencia = np.linspace(0, 50, n) # Tendencia alcista fuerte
    ruido = np.cumsum(np.random.randn(n) * 2) # Ruido y retrocesos
    precios = 1800 + tendencia + ruido # Precio base del Oro ~1800$
        
    df_test = pd.DataFrame({
        'open': precios,
        'high': precios + abs(np.random.normal(0, 5, n)),
        'low': precios - abs(np.random.normal(0, 5, n)),
        'close': precios
    })
    
    sistema = SistemaOro(capital_inicial=10000, riesgo_pct=1.0)
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
