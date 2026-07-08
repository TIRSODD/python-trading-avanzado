"""
checklist_validacion.py - Checklist automatizado de validación de backtests

Este módulo proporciona una lista de verificación (checklist) programática
para determinar si un backtest es estadísticamente robusto y fiable
antes de pasar a producción o a un forward test.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 4: Backtesting Robusto
"""

import pandas as pd


def validar_backtest(metricas):
    """
    Evalúa un diccionario de métricas de backtest contra unos criterios mínimos.
    
    Args:
        metricas (dict): Diccionario con las métricas del backtest.
            Debe contener: 'num_operaciones', 'profit_factor', 'sharpe_ratio', 
            'max_drawdown_pct', 'win_rate'.
    
    Returns:
        dict: Diccionario con el resultado de cada check (True/False) y un mensaje.
    """
    resultados = {}
    
    # 1. Número mínimo de operaciones (Significancia estadística)
    min_operaciones = 30
    check_ops = metricas.get('num_operaciones', 0) >= min_operaciones
    resultados['num_operaciones'] = {
        'aprobado': check_ops,
        'mensaje': f"{metricas.get('num_operaciones', 0)} operaciones (Mínimo: {min_operaciones})"
    }
    
    # 2. Profit Factor (Rentabilidad vs Pérdida)
    min_pf = 1.5
    pf = metricas.get('profit_factor', 0)
    check_pf = pf >= min_pf
    resultados['profit_factor'] = {
        'aprobado': check_pf,
        'mensaje': f"{pf:.2f} (Mínimo: {min_pf})"
    }
    
    # 3. Ratio de Sharpe (Rentabilidad ajustada al riesgo)
    min_sharpe = 1.0
    sharpe = metricas.get('sharpe_ratio', 0)
    check_sharpe = sharpe >= min_sharpe
    resultados['sharpe_ratio'] = {
        'aprobado': check_sharpe,
        'mensaje': f"{sharpe:.2f} (Mínimo: {min_sharpe})"
    }
    
    # 4. Máximo Drawdown (Caída máxima de la cuenta)
    max_dd_limite = 20.0
    max_dd = metricas.get('max_drawdown_pct', 100)
    check_dd = max_dd <= max_dd_limite
    resultados['max_drawdown'] = {
        'aprobado': check_dd,
        'mensaje': f"{max_dd:.2f}% (Máximo permitido: {max_dd_limite}%)"
    }
    
    # 5. Win Rate (Tasa de acierto - sanity check)
    # No hay un mínimo estricto, pero si es muy bajo (<30%) suele ser sospechoso 
    # a menos que el Profit Factor sea altísimo (estrategias de tendencia).
    win_rate = metricas.get('win_rate', 0) * 100
    check_wr = win_rate >= 30.0
    resultados['win_rate'] = {
        'aprobado': check_wr,
        'mensaje': f"{win_rate:.1f}% (Sanity check: > 30%)"
    }
    
    return resultados


def generar_informe(metricas):
    """
    Imprime un informe visual del checklist de validación.
    """
    print("=== CHECKLIST DE VALIDACIÓN DE BACKTEST ===\n")
    
    resultados = validar_backtest(metricas)
    aprobados = 0
    total = len(resultados)
    
    for concepto, datos in resultados.items():
        estado = "✅ APROBADO" if datos['aprobado'] else "❌ RECHAZADO"
        if datos['aprobado']:
            aprobados += 1
        print(f"[{estado}] {concepto.upper().replace('_', ' ')}: {datos['mensaje']}")
        
    print("\n" + "="*45)
    puntuacion = (aprobados / total) * 100
    print(f"PUNTUACIÓN: {aprobados}/{total} ({puntuacion:.0f}%)")
    
    if puntuacion == 100:
        print("🏆 ¡EXCELENTE! El backtest supera todos los criterios de robustez.")
    elif puntuacion >= 80:
        print("️ BUENO, pero revisa los puntos rechazados antes de operar en real.")
    else:
        print("🛑 NO VÁLIDO. El backtest no es fiable. No operes con esta estrategia.")
        
    return puntuacion == 100


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo checklist_validacion.py ===\n")
    
    # 1. Caso de éxito (Estrategia robusta)
    print("1. Evaluando Estrategia Robusta:")
    metricas_buenas = {
        'num_operaciones': 150,
        'profit_factor': 2.1,
        'sharpe_ratio': 1.8,
        'max_drawdown_pct': 12.5,
        'win_rate': 0.55
    }
    generar_informe(metricas_buenas)
    
    print("\n" + "="*50 + "\n")
    
    # 2. Caso de fracaso (Overfitting o mala estrategia)
    print("2. Evaluando Estrategia No Robusta (Overfitting):")
    metricas_malas = {
        'num_operaciones': 15, # Muy pocas operaciones
        'profit_factor': 4.5,  # Demasiado bueno para ser verdad
        'sharpe_ratio': 3.2,
        'max_drawdown_pct': 5.0,
        'win_rate': 0.90
    }
    generar_informe(metricas_malas)
    
    print("\n✓ Prueba completada exitosamente")
