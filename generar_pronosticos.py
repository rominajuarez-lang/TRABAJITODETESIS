import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

METODOS_PRONOSTICO = [
    "Naive",
    "Promedio móvil",
    "SES",
    "Regresión lineal",
    "ARIMA",
    "Holt-Winters",
    "Croston",
]

def asegurar_prediccion_valida(pred, serie) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pred.size == 0:
        return np.zeros(len(serie))
    valor_relleno = float(np.nanmean(serie)) if len(serie) else 0.0
    if np.isnan(valor_relleno):
        valor_relleno = 0.0
    pred = np.where(np.isfinite(pred), pred, valor_relleno)
    return np.maximum(0, pred)

def forecast_naive(serie: np.ndarray, pasos_futuros: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(serie) == 0:
        return np.array([]), np.array([])
    pred_hist = np.empty(len(serie), dtype=float)
    pred_hist[0] = serie[0]
    if len(serie) > 1:
        pred_hist[1:] = serie[:-1]
    ultimo = serie[-1]
    pred_future = np.repeat(ultimo, pasos_futuros)
    return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)

def forecast_promedio_movil(serie: np.ndarray, pasos_futuros: int = 0, ventana: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(serie) == 0:
        return np.array([]), np.array([])
    pred_hist = np.empty(len(serie), dtype=float)
    pred_hist[0] = serie[0]
    for i in range(1, len(serie)):
        inicio = max(0, i - ventana)
        pred_hist[i] = np.mean(serie[inicio:i])

    historial_extendido = list(serie.astype(float))
    futuros = []
    for _ in range(pasos_futuros):
        ultimos = historial_extendido[-ventana:]
        valor = float(np.mean(ultimos)) if ultimos else 0.0
        futuros.append(valor)
        historial_extendido.append(valor)
    return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, np.array(futuros))

def forecast_regresion(serie: np.ndarray, pasos_futuros: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(serie) == 0:
        return np.array([]), np.array([])
    
    if len(serie) < 12:
        return forecast_ses(serie, pasos_futuros)

    x = np.arange(len(serie)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, serie)
    pred_hist = modelo.predict(x)

    if pasos_futuros > 0:
        x_future = np.arange(len(serie), len(serie) + pasos_futuros).reshape(-1, 1)
        pred_future = modelo.predict(x_future)
        
        max_historico = np.max(serie)
        limite_superior = max_historico * 1.5 
        pred_future = np.clip(pred_future, a_min=0, a_max=limite_superior)
    else:
        pred_future = np.array([])

    return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)

def forecast_ses(serie: np.ndarray, pasos_futuros: int = 0, alpha: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    if len(serie) < 3:
        valor = float(np.mean(serie)) if len(serie) else 0.0
        return np.repeat(valor, len(serie)), np.repeat(valor, pasos_futuros)
    try:
        modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
        ajuste = modelo.fit(smoothing_level=alpha, optimized=False)
        pred_hist = np.asarray(ajuste.fittedvalues)
        pred_future = np.asarray(ajuste.forecast(pasos_futuros)) if pasos_futuros > 0 else np.array([])
        return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)
    except Exception:
        return forecast_promedio_movil(serie, pasos_futuros)

def forecast_arima(serie: np.ndarray, pasos_futuros: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # ESCUDO ANTI-CUELGUES: Si hay muchos ceros, ARIMA entra en bucle infinito. 
    if len(serie) < 6 or np.count_nonzero(serie) < (len(serie) * 0.4):
        return forecast_ses(serie, pasos_futuros)
    try:
        modelo = ARIMA(serie, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        ajuste = modelo.fit()
        pred_hist = np.asarray(ajuste.fittedvalues)
        pred_future = np.asarray(ajuste.forecast(pasos_futuros)) if pasos_futuros > 0 else np.array([])
        return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)
    except Exception:
        return forecast_ses(serie, pasos_futuros)

def forecast_holt_winters(serie: np.ndarray, pasos_futuros: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # ESCUDO ANTI-CUELGUES: Holt-Winters se cuelga intentando buscar estacionalidad en puros ceros
    if len(serie) < 18 or np.count_nonzero(serie) < (len(serie) * 0.4):
        return forecast_ses(serie, pasos_futuros)
    try:
        modelo = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
        ajuste = modelo.fit(optimized=True)
        pred_hist = np.asarray(ajuste.fittedvalues)
        pred_future = np.asarray(ajuste.forecast(pasos_futuros)) if pasos_futuros > 0 else np.array([])
        return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)
    except Exception:
        return forecast_ses(serie, pasos_futuros)

def forecast_croston(serie: np.ndarray, pasos_futuros: int = 0, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    serie = np.asarray(serie, dtype=float)
    n = len(serie)
    if n == 0:
        return np.array([]), np.array([])
    if np.all(serie == 0):
        return np.zeros(n), np.zeros(pasos_futuros)
    first_nonzero_idx = np.argmax(serie > 0)
    z = serie[first_nonzero_idx]
    p = first_nonzero_idx + 1 if first_nonzero_idx + 1 > 0 else 1
    q = z / p
    pred_hist = np.zeros(n, dtype=float)
    interval = 1
    for t in range(n):
        pred_hist[t] = q
        if serie[t] > 0:
            z = alpha * serie[t] + (1 - alpha) * z
            p = alpha * interval + (1 - alpha) * p
            q = z / max(p, 1e-9)
            interval = 1
        else:
            interval += 1
    pred_future = np.repeat(q, pasos_futuros)
    return asegurar_prediccion_valida(pred_hist, serie), np.maximum(0, pred_future)

def aplicar_metodo_pronostico(serie: np.ndarray, metodo: str, pasos_futuros: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if metodo == "Naive":
        return forecast_naive(serie, pasos_futuros)
    if metodo == "Promedio móvil":
        return forecast_promedio_movil(serie, pasos_futuros)
    if metodo == "Regresión lineal":
        return forecast_regresion(serie, pasos_futuros)
    if metodo == "ARIMA":
        return forecast_arima(serie, pasos_futuros)
    if metodo == "Holt-Winters":
        return forecast_holt_winters(serie, pasos_futuros)
    if metodo == "Croston":
        return forecast_croston(serie, pasos_futuros)
    return forecast_ses(serie, pasos_futuros)

def calcular_errores(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = y_real.sum()
    mae = np.mean(np.abs(y_real - y_pred)) if len(y_real) else 0.0
    rmse = np.sqrt(mean_squared_error(y_real, y_pred)) if len(y_real) else 0.0
    
    if suma_real == 0:
        return {"wMAPE": 0.0, "Bias": 0.0, "MAE": mae, "RMSE": rmse}
    
    wmape = np.sum(np.abs(y_real - y_pred)) / suma_real
    bias = np.sum(y_pred - y_real) / suma_real
    return {"wMAPE": wmape, "Bias": bias, "MAE": mae, "RMSE": rmse}

def calcular_meses_futuros(df: pd.DataFrame, fecha_fin) -> tuple[int, pd.Timestamp]:
    ultima_fecha = pd.to_datetime(df["date"].max()).to_period("M").to_timestamp()
    fecha_fin = pd.to_datetime(fecha_fin).to_period("M").to_timestamp()
    if fecha_fin <= ultima_fecha:
        return 0, ultima_fecha
    meses = (fecha_fin.year - ultima_fecha.year) * 12 + (fecha_fin.month - ultima_fecha.month)
    return int(meses), fecha_fin

def generar_fechas_futuras(ultima_fecha, pasos_futuros: int) -> pd.DatetimeIndex:
    if pasos_futuros <= 0:
        return pd.DatetimeIndex([])
    ultima_fecha = pd.to_datetime(ultima_fecha).to_period("M").to_timestamp()
    return pd.date_range(
        start=ultima_fecha + pd.offsets.MonthBegin(1),
        periods=pasos_futuros,
        freq="MS",
    )

def generar_forecast(df: pd.DataFrame, metodo: str, fecha_fin_pronostico=None) -> pd.DataFrame:
    resultados = []
    pasos_futuros, _ = calcular_meses_futuros(df, fecha_fin_pronostico) if fecha_fin_pronostico is not None else (0, None)
    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        pred_hist, pred_future = aplicar_metodo_pronostico(serie, metodo, pasos_futuros)
        err = calcular_errores(serie, pred_hist)
        sub["demand_forecast"] = np.round(pred_hist, 2)
        sub["method_used"] = metodo
        sub["method_wmape"] = err["wMAPE"]
        sub["method_bias"] = err["Bias"]
        sub["tipo_periodo"] = "Histórico"
        resultados.append(sub)
        if pasos_futuros > 0:
            fechas_futuras = generar_fechas_futuras(sub["date"].max(), pasos_futuros)
            futuro = pd.DataFrame(
                {
                    "date": fechas_futuras,
                    "product_id": producto,
                    "demand_real": np.round(pred_future, 2),
                    "demand_forecast": np.round(pred_future, 2),
                    "method_used": metodo,
                    "method_wmape": err["wMAPE"],
                    "method_bias": err["Bias"],
                    "tipo_periodo": "Pronóstico futuro",
                }
            )
            resultados.append(futuro)
    return pd.concat(resultados, ignore_index=True)

def generar_forecast_mejor_por_producto(df: pd.DataFrame, fecha_fin_pronostico=None):
    forecasts_finales = []
    comparacion = []
    pasos_futuros, _ = calcular_meses_futuros(df, fecha_fin_pronostico) if fecha_fin_pronostico is not None else (0, None)

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        
        n_total = len(serie)
        horizonte_test = max(2, min(6, int(n_total * 0.2))) 
        
        if n_total > horizonte_test * 2: 
            train = serie[:-horizonte_test]
            test = serie[-horizonte_test:]
        else:
            train = serie
            test = serie
            horizonte_test = 0

        predicciones_hist = {}
        predicciones_future = {}
        filas_producto = []

        for metodo in METODOS_PRONOSTICO:
            if horizonte_test > 0:
                _, pred_test = aplicar_metodo_pronostico(train, metodo, pasos_futuros=horizonte_test)
                err = calcular_errores(test, pred_test) 
            else:
                pred_hist_full, _ = aplicar_metodo_pronostico(serie, metodo, 0)
                err = calcular_errores(serie, pred_hist_full)

            pred_hist, pred_future = aplicar_metodo_pronostico(serie, metodo, pasos_futuros)
            
            predicciones_hist[metodo] = pred_hist
            predicciones_future[metodo] = pred_future
            
            fila = {
                "Producto": producto,
                "Método": metodo,
                "wMAPE": err["wMAPE"],
                "Bias": err["Bias"],
                "Abs_Bias": abs(err["Bias"]),
                "MAE": err["MAE"],
                "RMSE": err["RMSE"] 
            }
            comparacion.append(fila)
            filas_producto.append(fila)

        comp_producto = pd.DataFrame(filas_producto)
        mejor_fila = comp_producto.sort_values(["wMAPE", "RMSE", "Abs_Bias"]).iloc[0]
        mejor_metodo = mejor_fila["Método"]
        
        sub["demand_forecast"] = np.round(predicciones_hist[mejor_metodo], 2)
        sub["method_used"] = mejor_metodo
        sub["method_wmape"] = float(mejor_fila["wMAPE"])
        sub["method_bias"] = float(mejor_fila["Bias"])
        sub["tipo_periodo"] = "Histórico"
        forecasts_finales.append(sub)

        if pasos_futuros > 0:
            fechas_futuras = generar_fechas_futuras(sub["date"].max(), pasos_futuros)
            futuro = pd.DataFrame(
                {
                    "date": fechas_futuras,
                    "product_id": producto,
                    "demand_real": np.round(predicciones_future[mejor_metodo], 2),
                    "demand_forecast": np.round(predicciones_future[mejor_metodo], 2),
                    "method_used": mejor_metodo,
                    "method_wmape": float(mejor_fila["wMAPE"]),
                    "method_bias": float(mejor_fila["Bias"]),
                    "tipo_periodo": "Pronóstico futuro",
                }
            )
            forecasts_finales.append(futuro)

    df_comparacion = pd.DataFrame(comparacion)
    mejores = (
        df_comparacion.sort_values(["Producto", "wMAPE", "RMSE", "Abs_Bias"])
        .groupby("Producto", as_index=False)
        .first()[["Producto", "Método"]]
        .rename(columns={"Método": "Mejor método"})
    )

    df_comparacion = df_comparacion.merge(mejores, on="Producto", how="left")
    df_comparacion["Es mejor"] = df_comparacion["Método"] == df_comparacion["Mejor método"]
    df_comparacion = df_comparacion.drop(columns=["Abs_Bias", "RMSE"]) 

    return pd.concat(forecasts_finales, ignore_index=True), df_comparacion
