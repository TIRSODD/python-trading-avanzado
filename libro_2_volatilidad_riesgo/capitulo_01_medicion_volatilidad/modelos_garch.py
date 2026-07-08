"""
modelos_garch.py - Modelos GARCH para predicción de volatilidad

Este módulo implementa modelos GARCH(1,1) para capturar la persistencia 
de la volatilidad en series financieras.

Autor: TIRSODD
Libro: Python para Trading Algorítmico - Libro 2
Capítulo 1: Medición y Modelado de Volatilidad
"""

import numpy as np
import pandas as pd

# Nota: Requiere la librería 'arch' instalada (pip install arch)
try:
    from arch import arch_model
except ImportError:
    print("⚠️ Aviso: La librería 'arch' no está instalada. Instálala con: pip install arch")


def modelo_garch_basico(df, periodo=14):
    """
    Ajusta un modelo GARCH(1,1) básico a los log-retornos del precio.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'close'.
        periodo (int): Ventana de datos a usar (default: 14).
    
    Returns:
        object: Resultado del modelo ajustado (arch model fit).
    
    Ejemplo:
        >>> resultado = modelo_garch_basico(df)
        >>> print(resultado.params)
    """
    # Calcular log-retornos
    df['log_retorno'] = np.log(df['close'] / df['close'].shift(1))
    df_retornos = df['log_retorno'].dropna() * 100  # Escalar para mejor convergencia
    
    if len(df_retornos) < periodo:
        raise ValueError(f"Datos insuficientes. Se necesitan al menos {periodo} períodos.")
    
    # Ajustar modelo GARCH(1,1)
    # vol='Garch', p=1 (componente AR), q=1 (componente MA)
    modelo = arch_model(df_retornos, vol='Garch', p=1, q=1, dist='normal')
    resultado = modelo.fit(disp='off')
    
    # Interpretación de parámetros
    params = resultado.params
    
    print("=== MODELO GARCH(1,1) ===\n")
    print(f"Omega (ω): {params['Omega']:.6f} → Volatilidad base")
    print(f"Alpha (α): {params['alpha[1]']:.6f} → Impacto de shocks recientes")
    print(f"Beta (β): {params['beta[1]']:.6f} → Persistencia de la volatilidad")
    
    persistencia = params['alpha[1]'] + params['beta[1]']
    print(f"\nPersistencia (α + β): {persistencia:.4f}")
    
    if persistencia > 0.9:
        print("→ Alta persistencia: los shocks de volatilidad duran mucho tiempo.")
    else:
        print("→ Baja persistencia: la volatilidad vuelve rápido a la normalidad.")
    
    return resultado


def predecir_volatilidad_garch(resultado_garch, horizonte=5):
    """
    Predice la volatilidad para los próximos N períodos usando un modelo GARCH ajustado.
    
    Args:
        resultado_garch: Objeto resultado de arch_model.fit().
        horizonte (int): Número de períodos a predecir.
    
    Returns:
        np.array: Array con las volatilidades predichas.
    """
    predicciones = resultado_garch.forecast(horizon=horizonte)
    varianza_predicha = predicciones.variance.iloc[-1].values
    
    # La volatilidad es la raíz cuadrada de la varianza
    volatilidad_predicha = np.sqrt(varianza_predicha)
    
    print(f"\n=== PREDICCIÓN DE VOLATILIDAD ({horizonte} períodos) ===\n")
    for i, vol in enumerate(volatilidad_predicha, 1):
        print(f"  Período +{i}: Volatilidad predicha = {vol:.4f}%")
    
    return volatilidad_predicha


def comparar_garch_vs_historico(df, periodo_hv=20):
    """
    Compara el error de predicción entre GARCH y la Volatilidad Histórica simple.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'close'.
        periodo_hv (int): Ventana para la volatilidad histórica.
    
    Returns:
        tuple: (error_medio_garch, error_medio_hv)
    """
    df['log_retorno'] = np.log(df['close'] / df['close'].shift(1))
    df['hv'] = df['log_retorno'].rolling(periodo_hv).std() * 100
    
    # Usamos una ventana rolling para simular predicciones en el tiempo
    ventana_rolling = 100
    errores_garch = []
    errores_hv = []
    
    # Iteramos sobre los datos (saltando los primeros para tener suficiente historial)
    for i in range(ventana_rolling, len(df) - 1):
        retornos_train = df['log_retorno'].iloc[i-ventana_rolling:i].dropna() * 100
        
        if len(retornos_train) < 50:
            continue
        
        try:
            # Ajustar GARCH con datos hasta el momento 'i'
            modelo = arch_model(retornos_train, vol='Garch', p=1, q=1, dist='normal')
            resultado = modelo.fit(disp='off')
            
            # Predicción para el siguiente período
            pred = resultado.forecast(horizon=1)
            var_pred = pred.variance.iloc[-1].values[0]
            vol_garch_pred = np.sqrt(var_pred)
            
            # Volatilidad Histórica simple
            vol_hv_pred = df['hv'].iloc[i]
            
            # Volatilidad real del siguiente período (valor absoluto del retorno)
            vol_real = abs(df['log_retorno'].iloc[i+1]) * 100
            
            errores_garch.append(abs(vol_garch_pred - vol_real))
            errores_hv.append(abs(vol_hv_pred - vol_real))
        except Exception:
            continue
    
    if not errores_garch:
        return 0.0, 0.0
        
    error_medio_garch = np.mean(errores_garch)
    error_medio_hv = np.mean(errores_hv)
    
    print("\n=== COMPARACIÓN: GARCH vs. VOLATILIDAD HISTÓRICA ===\n")
    print(f"Error medio GARCH: {error_medio_garch:.4f}")
    print(f"Error medio HV:    {error_medio_hv:.4f}")
    
    if error_medio_hv > 0:
        mejora = 100 * (error_medio_hv - error_medio_garch) / error_medio_hv
        print(f"Mejora relativa de GARCH: {mejora:.1f}%")
    
    return error_medio_garch, error_medio_hv


# ============================================
# Código de prueba
# ============================================
if __name__ == "__main__":
    print("=== Probando módulo modelos_garch.py ===\n")
    
    # Generar datos sintéticos con clusters de volatilidad
    np.random.seed(42)
    n = 500
    volatilidad = np.zeros(n)
    retornos = np.zeros(n)
    
    for i in range(1, n):
        volatilidad[i] = 0.05 + 0.9 * volatilidad[i-1] + 0.05 * np.abs(np.random.randn())
        retornos[i] = np.random.randn() * volatilidad[i]
        
    df_ejemplo = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(retornos / 100))
    })
    
    # Probar modelo básico
    print("1. Ajustando modelo GARCH...")
    resultado = modelo_garch_basico(df_ejemplo)
    
    # Probar predicción
    print("\n2. Prediciendo volatilidad futura...")
    predecir_volatilidad_garch(resultado, horizonte=3)
    
    print("\n✓ Prueba completada exitosamente")
