"""
checklist_produccion.py - Checklist final antes de pasar a Producción (Real)

Este módulo contiene la lista de verificación definitiva que debe aprobar 
una estrategia antes de conectarla a un broker real. Evalúa métricas, 
costes de transacción y calidad de los datos.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 5: Caso Práctico - EUR/GBP
"""

import pandas as pd
import numpy as np


class ChecklistProduccion:
    """
    Clase que gestiona la validación final de una estrategia para su paso a real.
    """
    
    def __init__(self):
        self.resultados = []
        
    def verificar_metricas(self, metricas):
        """
        Verifica que las métricas de backtesting cumplan los mínimos exigidos.
        """
        print("--- 1. Verificación de Métricas ---")
        
        checks = [
            ('Rentabilidad', metricas.get('pnl_total', 0) > 0),
            ('Profit Factor > 1.5', metricas.get('profit_factor', 0) >= 1.5),
            ('Sharpe Ratio > 1.0', metricas.get('sharpe_ratio', 0) >= 1.0),
            ('Max Drawdown < 15%', metricas.get('max_drawdown_pct', 100) <= 15.0),
            ('Número de operaciones > 100', metricas.get('num_operaciones', 0) >= 100)
        ]
        
        for nombre, estado in checks:
            self._registrar(nombre, estado)
            
    def verificar_costes(self, metricas):
        """
        Verifica si la estrategia es rentable después de aplicar costes reales.
        """
        print("\n--- 2. Verificación de Costes (Slippage y Comisiones) ---")
        
        # Asumimos un coste medio de 1.5 pips por operación (comisión + spread + slippage)
        coste_pips = 1.5
        num_ops = metricas.get('num_operaciones', 0)
        beneficio_pips = metricas.get('beneficio_medio_pips', 0)
        
        # Si el beneficio medio por operación no cubre los costes, es un problema
        es_rentable_con_costes = beneficio_pips > coste_pips
        
        self._registrar(f'Beneficio medio ({beneficio_pips:.1f} pips) > Costes ({coste_pips} pips)', es_rentable_con_costes)
        
    def verificar_calidad_datos(self, df_datos):
        """
        Verifica que los datos de entrada no tengan errores graves.
        """
        print("\n--- 3. Verificación de Calidad de Datos ---")
        
        tiene_nans = df_datos.isnull().any().any()
        self._registrar('Datos sin valores nulos (NaN)', not tiene_nans)
        
        # Comprobar que los precios no son negativos ni cero
        precios_invalidos = (df_datos[['high', 'low', 'close']] <= 0).any().any()
        self._registrar('Precios siempre positivos', not precios_invalidos)
        
        # Comprobar coherencia High >= Low
        incoherencia = (df_datos['high'] < df_datos['low']).any()
        self._registrar('Coherencia High >= Low', not incoherencia)

    def _registrar(self, concepto, aprobado):
        """
        Registra el resultado de un check y lo imprime.
        """
        estado = "✅ OK" if aprobado else "❌ FALLO"
        print(f"[{estado}] {concepto}")
        self.resultados.append(aprobado)
        
    def informe_final(self):
        """
        Muestra el veredicto final.
        """
        print("\n" + "="*40)
        total = len(self.resultados)
        aprobados = sum(self.resultados)
        print(f"RESULTADO FINAL: {aprobados}/{total} checks superados.")
        
        if all(self.resultados):
            print("🚀 ¡APROBADO! La estrategia está lista para pasar a Producción (Forward Test / Real).")
        else:
            print("🛑 RECHAZADO. La estrategia NO está lista. Revisa los puntos fallidos.")
        print("="*40)


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando Checklist de Producción ===\n")
    
    # 1. Simular métricas de una estrategia buena
    metricas_buenas = {
        'pnl_total': 1500.0,
        'profit_factor': 1.8,
        'sharpe_ratio': 1.5,
        'max_drawdown_pct': 12.0,
        'num_operaciones': 150,
        'beneficio_medio_pips': 4.5
    }
    
    # 2. Generar datos sintéticos limpios
    np.random.seed(42)
    n = 500
    df_limpio = pd.DataFrame({
        'open': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005),
        'high': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005) + 0.0010,
        'low': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005) - 0.0010,
        'close': 0.8500 + np.cumsum(np.random.randn(n) * 0.0005)
    })
    
    # Ejecutar el checklist
    checklist = ChecklistProduccion()
    checklist.verificar_metricas(metricas_buenas)
    checklist.verificar_costes(metricas_buenas)
    checklist.verificar_calidad_datos(df_limpio)
    
    checklist.informe_final()
    
    print("\n✓ Prueba completada exitosamente")
