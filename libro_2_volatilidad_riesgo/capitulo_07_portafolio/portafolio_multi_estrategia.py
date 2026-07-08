"""
portafolio_multi_estrategia.py - Construcción de un Portafolio Multi-Estrategia

Este módulo contiene la clase principal para combinar diferentes estrategias 
(EUR/GBP, Oro, Plata) en un único portafolio, calculando las métricas 
globales de rendimiento y riesgo.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 7: Construcción de un Portafolio Multi-Estrategia
"""

import numpy as np
import pandas as pd


class PortafolioMultiEstrategia:
    """
    Clase para gestionar y combinar múltiples estrategias de trading.
    """
    
    def __init__(self):
        self.estrategias = {}
        self.equity_combinada = None
        
    def agregar_estrategia(self, nombre, equity_curve):
        """
        Añade la curva de equity de una estrategia al portafolio.
        
        Args:
            nombre (str): Nombre identificador de la estrategia (ej: 'EURGBP', 'ORO').
            equity_curve (pd.Series o list): Curva de equity diaria de la estrategia.
        """
        self.estrategias[nombre] = pd.Series(equity_curve)
        print(f"✓ Estrategia '{nombre}' añadida al portafolio.")
        
    def calcular_equity_combinada(self, pesos=None):
        """
        Calcula la curva de equity combinada del portafolio.
        
        Args:
            pesos (dict): Diccionario con el peso de cada estrategia (ej: {'EURGBP': 0.5, 'ORO': 0.5}).
                          Si es None, se asume reparto equitativo.
        """
        if not self.estrategias:
            print("⚠️ No hay estrategias en el portafolio.")
            return
            
        # Si no se pasan pesos, asignar pesos iguales
        if pesos is None:
            num_estrategias = len(self.estrategias)
            pesos = {nombre: 1.0 / num_estrategias for nombre in self.estrategias.keys()}
            
        # Unir todas las curvas de equity en un DataFrame
        df_equity = pd.DataFrame(self.estrategias)
        
        # Normalizar las curvas para que empiecen en 1 (para poder sumarlas correctamente)
        df_normalizado = df_equity / df_equity.iloc[0]
        
        # Calcular la equity combinada ponderada
        equity_combinada = pd.Series(0.0, index=df_normalizado.index)
        for nombre, peso in pesos.items():
            if nombre in df_normalizado.columns:
                equity_combinada += df_normalizado[nombre] * peso
                
        # Escalar al capital inicial (asumimos 10000 para el ejemplo)
        capital_inicial = 10000
        self.equity_combinada = equity_combinada * capital_inicial
        
        print("\n✓ Curva de equity combinada calculada.")
        
    def calcular_metricas_portafolio(self):
        """
        Calcula las métricas globales del portafolio combinado.
        """
        if self.equity_combinada is None:
            self.calcular_equity_combinada()
            
        equity = self.equity_combinada
        retornos_diarios = equity.pct_change().dropna()
        
        # Métricas básicas
        rentabilidad_total = (equity.iloc[-1] / equity.iloc[0]) - 1
        rentabilidad_anualizada = (1 + rentabilidad_total) ** (252 / len(equity)) - 1
        
        volatilidad_anualizada = retornos_diarios.std() * np.sqrt(252)
        sharpe_ratio = (rentabilidad_anualizada - 0.02) / volatilidad_anualizada if volatilidad_anualizada > 0 else 0
        
        # Drawdown máximo
        pico_acumulado = equity.cummax()
        drawdown = (equity - pico_acumulado) / pico_acumulado
        max_drawdown = drawdown.min()
        
        print("\n=== MÉTRICAS DEL PORTAFOLIO COMBINADO ===")
        print(f"Rentabilidad Total:   {rentabilidad_total:.2%}")
        print(f"Rentabilidad Anual:   {rentabilidad_anualizada:.2%}")
        print(f"Volatilidad Anual:    {volatilidad_anualizada:.2%}")
        print(f"Ratio de Sharpe:      {sharpe_ratio:.2f}")
        print(f"Máximo Drawdown:      {max_drawdown:.2%}")
        
        return {
            'rentabilidad_total': rentabilidad_total,
            'rentabilidad_anualizada': rentabilidad_anualizada,
            'volatilidad_anualizada': volatilidad_anualizada,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando Portafolio Multi-Estrategia ===\n")
    
    # Generar curvas de equity sintéticas para 3 estrategias
    np.random.seed(42)
    dias = 252
    
    # Estrategia 1: EUR/GBP (Baja volatilidad, retornos estables)
    equity_eurgbp = 10000 + np.cumsum(np.random.normal(5, 15, dias))
    
    # Estrategia 2: Oro (Mayor volatilidad, tendencia alcista)
    equity_oro = 10000 + np.cumsum(np.random.normal(10, 30, dias))
    
    # Estrategia 3: Plata (Alta volatilidad, más ruido)
    equity_plata = 10000 + np.cumsum(np.random.normal(8, 45, dias))
    
    # Crear portafolio y añadir estrategias
    portafolio = PortafolioMultiEstrategia()
    portafolio.agregar_estrategia('EURGBP', equity_eurgbp)
    portafolio.agregar_estrategia('ORO', equity_oro)
    portafolio.agregar_estrategia('PLATA', equity_plata)
    
    # Calcular equity combinada (pesos iguales: 33.3% cada una)
    portafolio.calcular_equity_combinada()
    
    # Calcular y mostrar métricas
    metricas = portafolio.calcular_metricas_portafolio()
    
    print("\n✓ Prueba completada exitosamente")
