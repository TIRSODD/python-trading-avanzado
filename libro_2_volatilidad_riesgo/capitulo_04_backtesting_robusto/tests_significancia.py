"""
tests_significancia.py - Tests de significancia estadística para backtesting

Este módulo implementa pruebas estadísticas para determinar si los resultados
de un backtest son estadísticamente significativos o si podrían haberse 
obtenido por pura suerte (azar).

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 4: Backtesting Robusto
"""

import numpy as np
import pandas as pd
from scipy import stats


def t_test_rentabilidad(retornos_diarios, tasa_libre_riesgo=0.02):
    """
    Realiza una prueba t de Student para ver si la rentabilidad media 
    es estadísticamente mayor que cero (o que la tasa libre de riesgo).
    
    Args:
        retornos_diarios (list o pd.Series): Serie de retornos diarios.
        tasa_libre_riesgo (float): Tasa anualizada libre de riesgo.
    
    Returns:
        dict: Estadístico t, p-valor y conclusión.
    """
    retornos = np.array(retornos_diarios)
    # Ajustar la tasa libre de riesgo a diaria
    tasa_diaria = tasa_libre_riesgo / 252
    
    # Calcular retornos en exceso
    retornos_exceso = retornos - tasa_diaria
    
    # Realizar el test t de una muestra (H0: media = 0)
    t_stat, p_valor = stats.ttest_1samp(retornos_exceso, 0)
    
    # Conclusión
    if p_valor < 0.05:
        conclusion = "✓ Significativo: La rentabilidad es estadísticamente mayor que cero."
    else:
        conclusion = "⚠️ No significativo: No hay evidencia suficiente de que la estrategia gane dinero."
        
    return {
        't_statistic': t_stat,
        'p_valor': p_valor,
        'conclusion': conclusion
    }


def bootstrap_significancia_sharpe(retornos_diarios, num_simulaciones=10000, confianza=0.95):
    """
    Usa bootstrapping para calcular el intervalo de confianza del Ratio de Sharpe.
    
    Si el intervalo de confianza NO incluye el cero, el Sharpe es significativo.
    
    Args:
        retornos_diarios (list o pd.Series): Serie de retornos diarios.
        num_simulaciones (int): Número de muestras bootstrap.
        confianza (float): Nivel de confianza (ej: 0.95 para 95%).
    
    Returns:
        dict: Sharpe original, intervalo de confianza y conclusión.
    """
    retornos = np.array(retornos_diarios)
    
    # Calcular Sharpe real
    sharpe_real = (retornos.mean() / retornos.std()) * np.sqrt(252)
    
    # Simulaciones Bootstrap
    sharpes_simulados = []
    for _ in range(num_simulaciones):
        # Muestreo con reemplazo
        muestra = np.random.choice(retornos, size=len(retornos), replace=True)
        sharpe_sim = (muestra.mean() / muestra.std()) * np.sqrt(252)
        sharpes_simulados.append(sharpe_sim)
        
    # Calcular percentiles
    alpha = 1 - confianza
    percentil_bajo = np.percentile(sharpes_simulados, (alpha / 2) * 100)
    percentil_alto = np.percentile(sharpes_simulados, (1 - alpha / 2) * 100)
    
    # Conclusión
    if percentil_bajo > 0:
        conclusion = "✓ Sharpe significativo: El intervalo de confianza no incluye el cero."
    else:
        conclusion = "⚠️ Sharpe no significativo: El intervalo incluye el cero (podría ser suerte)."
        
    return {
        'sharpe_real': sharpe_real,
        'ic_bajo': percentil_bajo,
        'ic_alto': percentil_alto,
        'conclusion': conclusion
    }


def calcular_probabilidad_exito(num_ganadoras, num_total_operaciones):
    """
    Calcula la probabilidad de que una tasa de aciertos (win rate) sea 
    significativamente mayor que el 50% (azar).
    
    Usa la distribución binomial.
    
    Args:
        num_ganadoras (int): Número de operaciones ganadoras.
        num_total_operaciones (int): Número total de operaciones.
    
    Returns:
        dict: Win rate, p-valor y conclusión.
    """
    win_rate = num_ganadoras / num_total_operaciones
    
    # Test binomial: probabilidad de obtener X o más éxitos si p=0.5
    p_valor = stats.binom_test(num_ganadoras, num_total_operaciones, p=0.5, alternative='greater')
    
    if p_valor < 0.05:
        conclusion = "✓ Win rate significativo: Mayor que el azar (50%)."
    else:
        conclusion = "⚠️ Win rate no significativo: Podría ser producto del azar."
        
    return {
        'win_rate': win_rate,
        'p_valor': p_valor,
        'conclusion': conclusion
    }


def informe_significancia_completo(retornos_diarios, num_ganadoras, num_total):
    """
    Genera un informe completo con todos los tests de significancia.
    """
    print("=== INFORME DE SIGNIFICANCIA ESTADÍSTICA ===\n")
    
    print("1. Test de Rentabilidad (t-Student):")
    res_t = t_test_rentabilidad(retornos_diarios)
    print(f"   p-valor: {res_t['p_valor']:.4f}")
    print(f"   {res_t['conclusion']}\n")
    
    print("2. Significancia del Ratio de Sharpe (Bootstrap):")
    res_boot = bootstrap_significancia_sharpe(retornos_diarios)
    print(f"   Sharpe real: {res_boot['sharpe_real']:.2f}")
    print(f"   IC 95%: [{res_boot['ic_bajo']:.2f}, {res_boot['ic_alto']:.2f}]")
    print(f"   {res_boot['conclusion']}\n")
    
    print("3. Significancia del Win Rate (Binomial):")
    res_bin = calcular_probabilidad_exito(num_ganadoras, num_total)
    print(f"   Win rate: {res_bin['win_rate']:.2%}")
    print(f"   p-valor: {res_bin['p_valor']:.4f}")
    print(f"   {res_bin['conclusion']}\n")


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo tests_significancia.py ===\n")
    
    # Generar datos sintéticos con un sesgo positivo (estrategia ganadora)
    np.random.seed(42)
    retornos_ganadores = np.random.normal(0.0015, 0.01, 252)
    
    # Generar datos sintéticos sin sesgo (estrategia perdedora/azar)
    retornos_azar = np.random.normal(0.0, 0.01, 252)
    
    print("1. Probando con estrategia ganadora:")
    informe_significancia_completo(retornos_ganadores, num_ganadoras=140, num_total=252)
    
    print("="*50 + "\n")
    
    print("2. Probando con estrategia al azar:")
    informe_significancia_completo(retornos_azar, num_ganadoras=126, num_total=252)
    
    print("✓ Prueba completada exitosamente")
