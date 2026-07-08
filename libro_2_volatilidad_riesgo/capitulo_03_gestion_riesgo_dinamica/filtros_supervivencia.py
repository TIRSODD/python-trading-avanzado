"""
filtros_supervivencia.py - Filtros de supervivencia y Circuit Breakers

Este módulo implementa mecanismos de seguridad para detectar cuándo una 
estrategia de trading ha dejado de funcionar y debe detenerse automáticamente.

Es una de las herramientas más importantes para proteger el capital.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 3: Gestión de Riesgo Dinámica
"""

import numpy as np
import pandas as pd


class FiltroDrawdown:
    """
    Filtro que detiene la estrategia si el drawdown supera un umbral máximo.
    
    El drawdown es la caída desde el máximo histórico de la curva de equity.
    """
    
    def __init__(self, max_drawdown_pct=10.0):
        """
        Args:
            max_drawdown_pct (float): Drawdown máximo permitido en porcentaje.
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = 0
        self.drawdown_actual = 0
        self.estrategia_detenida = False
        
    def actualizar(self, equity_actual):
        """
        Actualiza el filtro con el nuevo valor de equity.
        
        Args:
            equity_actual (float): Valor actual de la curva de equity.
        
        Returns:
            bool: True si la estrategia debe detenerse, False si puede continuar.
        """
        # Actualizar el pico histórico
        if equity_actual > self.peak_equity:
            self.peak_equity = equity_actual
            
        # Calcular drawdown actual
        if self.peak_equity > 0:
            self.drawdown_actual = ((self.peak_equity - equity_actual) / self.peak_equity) * 100
        else:
            self.drawdown_actual = 0
            
        # Comprobar si se supera el umbral
        if self.drawdown_actual >= self.max_drawdown_pct:
            self.estrategia_detenida = True
            return True  # Detener estrategia
            
        return False  # Continuar operando
    
    def get_estado(self):
        """Retorna el estado actual del filtro."""
        return {
            'drawdown_actual': round(self.drawdown_actual, 2),
            'max_drawdown_permitido': self.max_drawdown_pct,
            'estrategia_detenida': self.estrategia_detenida
        }


class FiltroRachaPerdidas:
    """
    Filtro que detiene la estrategia si hay demasiadas pérdidas consecutivas.
    
    Una racha de pérdidas puede indicar que el mercado ha cambiado de régimen
    y la estrategia ya no es efectiva.
    """
    
    def __init__(self, max_perdidas_consecutivas=5):
        """
        Args:
            max_perdidas_consecutivas (int): Número máximo de pérdidas seguidas.
        """
        self.max_perdidas = max_perdidas_consecutivas
        self.racha_actual = 0
        self.estrategia_detenida = False
        
    def actualizar(self, pnl_operacion):
        """
        Actualiza el filtro con el resultado de la última operación.
        
        Args:
            pnl_operacion (float): Profit/Loss de la operación (positivo = ganancia, negativo = pérdida).
        
        Returns:
            bool: True si la estrategia debe detenerse.
        """
        if pnl_operacion < 0:
            # Pérdida: incrementar racha
            self.racha_actual += 1
        else:
            # Ganancia: resetear racha
            self.racha_actual = 0
            
        # Comprobar si se supera el límite
        if self.racha_actual >= self.max_perdidas:
            self.estrategia_detenida = True
            return True
            
        return False
    
    def get_estado(self):
        """Retorna el estado actual del filtro."""
        return {
            'racha_perdidas_actual': self.racha_actual,
            'max_perdidas_permitidas': self.max_perdidas,
            'estrategia_detenida': self.estrategia_detenida
        }


class FiltroDegradacionRendimiento:
    """
    Filtro que detecta si el rendimiento reciente es significativamente peor
    que el rendimiento histórico esperado.
    
    Usa una ventana móvil para comparar el rendimiento reciente vs el histórico.
    """
    
    def __init__(self, ventana_reciente=20, degradacion_minima_pct=50.0):
        """
        Args:
            ventana_reciente (int): Número de operaciones recientes a analizar.
            degradacion_minima_pct (float): Porcentaje mínimo de degradación para activar el filtro.
        """
        self.ventana_reciente = ventana_reciente
        self.degradacion_minima = degradacion_minima_pct
        self.historial_pnl = []
        self.estrategia_detenida = False
        
    def actualizar(self, pnl_operacion):
        """
        Actualiza el filtro con el resultado de la última operación.
        
        Args:
            pnl_operacion (float): Profit/Loss de la operación.
        
        Returns:
            bool: True si la estrategia debe detenerse.
        """
        self.historial_pnl.append(pnl_operacion)
        
        # Necesitamos suficientes datos para comparar
        if len(self.historial_pnl) < self.ventana_reciente * 2:
            return False
            
        # Calcular rendimiento histórico (todas las operaciones excepto las recientes)
        pnl_historico = self.historial_pnl[:-self.ventana_reciente]
        pnl_reciente = self.historial_pnl[-self.ventana_reciente:]
        
        media_historica = np.mean(pnl_historico)
        media_reciente = np.mean(pnl_reciente)
        
        # Calcular degradación
        if media_historica > 0:
            degradacion = ((media_historica - media_reciente) / abs(media_historica)) * 100
        else:
            degradacion = 0
            
        # Comprobar si la degradación es significativa
        if degradacion >= self.degradacion_minima:
            self.estrategia_detenida = True
            return True
            
        return False
    
    def get_estado(self):
        """Retorna el estado actual del filtro."""
        if len(self.historial_pnl) < self.ventana_reciente * 2:
            return {
                'estado': 'Datos insuficientes para evaluar',
                'operaciones_totales': len(self.historial_pnl)
            }
            
        pnl_historico = self.historial_pnl[:-self.ventana_reciente]
        pnl_reciente = self.historial_pnl[-self.ventana_reciente:]
        
        return {
            'pnl_medio_historico': round(np.mean(pnl_historico), 2),
            'pnl_medio_reciente': round(np.mean(pnl_reciente), 2),
            'operaciones_totales': len(self.historial_pnl),
            'estrategia_detenida': self.estrategia_detenida
        }


class CircuitBreaker:
    """
    Circuit Breaker maestro que combina todos los filtros de supervivencia.
    
    Si CUALQUIERA de los filtros se activa, la estrategia se detiene.
    """
    
    def __init__(self, max_drawdown=10.0, max_perdidas_consecutivas=5, 
                 degradacion_rendimiento=50.0):
        """
        Args:
            max_drawdown (float): Drawdown máximo en porcentaje.
            max_perdidas_consecutivas (int): Máximo de pérdidas seguidas.
            degradacion_rendimiento (float): Degradación mínima de rendimiento.
        """
        self.filtro_drawdown = FiltroDrawdown(max_drawdown)
        self.filtro_racha = FiltroRachaPerdidas(max_perdidas_consecutivas)
        self.filtro_degradacion = FiltroDegradacionRendimiento(20, degradacion_rendimiento)
        
        self.estrategia_detenida = False
        self.motivo_detencion = None
        
    def actualizar(self, equity_actual, pnl_operacion):
        """
        Actualiza todos los filtros con los nuevos datos.
        
        Args:
            equity_actual (float): Valor actual de la curva de equity.
            pnl_operacion (float): PnL de la última operación.
        
        Returns:
            bool: True si la estrategia debe detenerse.
        """
        if self.estrategia_detenida:
            return True  # Ya está detenida, no hacer nada
            
        # Actualizar cada filtro
        if self.filtro_drawdown.actualizar(equity_actual):
            self.estrategia_detenida = True
            self.motivo_detencion = f"Drawdown máximo superado: {self.filtro_drawdown.drawdown_actual:.2f}%"
            return True
            
        if self.filtro_racha.actualizar(pnl_operacion):
            self.estrategia_detenida = True
            self.motivo_detencion = f"Racha de pérdidas: {self.filtro_racha.racha_actual} pérdidas consecutivas"
            return True
            
        if self.filtro_degradacion.actualizar(pnl_operacion):
            self.estrategia_detenida = True
            self.motivo_detencion = "Degradación significativa del rendimiento"
            return True
            
        return False
    
    def get_estado_completo(self):
        """Retorna el estado de todos los filtros."""
        return {
            'estrategia_detenida': self.estrategia_detenida,
            'motivo_detencion': self.motivo_detencion,
            'filtro_drawdown': self.filtro_drawdown.get_estado(),
            'filtro_racha': self.filtro_racha.get_estado(),
            'filtro_degradacion': self.filtro_degradacion.get_estado()
        }


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo filtros_supervivencia.py ===\n")
    
    # 1. Prueba de Filtro de Drawdown
    print("1. Filtro de Drawdown:")
    filtro_dd = FiltroDrawdown(max_drawdown_pct=10.0)
    equity = [10000, 10100, 10200, 10150, 9500, 9400]  # Caída del 7.8%
    
    for eq in equity:
        detenido = filtro_dd.actualizar(eq)
        if detenido:
            print(f"   ⚠️ ESTRATEGIA DETENIDA en equity={eq}")
            break
        else:
            print(f"   Equity={eq}, Drawdown={filtro_dd.drawdown_actual:.2f}%")
    
    print()
    
    # 2. Prueba de Filtro de Racha de Pérdidas
    print("2. Filtro de Racha de Pérdidas:")
    filtro_racha = FiltroRachaPerdidas(max_perdidas_consecutivas=3)
    pnls = [100, -50, -75, -100, 200]  # 3 pérdidas consecutivas
    
    for pnl in pnls:
        detenido = filtro_racha.actualizar(pnl)
        if detenido:
            print(f"   ⚠️ ESTRATEGIA DETENIDA tras PnL={pnl}")
            break
        else:
            print(f"   PnL={pnl}, Racha={filtro_racha.racha_actual}")
    
    print()
    
    # 3. Prueba de Circuit Breaker completo
    print("3. Circuit Breaker completo:")
    cb = CircuitBreaker(max_drawdown=10.0, max_perdidas_consecutivas=3, degradacion_rendimiento=50.0)
    
    # Simular operaciones
    equity_sim = [10000]
    pnls_sim = [100, 150, -50, 200, -75, -100, -150]
    
    for i, pnl in enumerate(pnls_sim):
        equity_actual = equity_sim[-1] + pnl
        equity_sim.append(equity_actual)
        
        detenido = cb.actualizar(equity_actual, pnl)
        print(f"   Operación {i+1}: PnL={pnl}, Equity={equity_actual}")
        
        if detenido:
            print(f"   ⚠️ ESTRATEGIA DETENIDA: {cb.motivo_detencion}")
            break
    
    print("\n✓ Prueba completada exitosamente")
