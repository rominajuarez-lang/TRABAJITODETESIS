import numpy as np
import pandas as pd

def convertir_a_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte cualquier base diaria/semanal/mensual a demanda mensual por producto."""
    df = df.copy()
    
    # 1. Asegurar formato de fechas y convertir demandas erróneas o negativas a 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["demand_real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["demand_real"] = df["demand_real"].clip(lower=0)
    df = df.dropna(subset=["date"])

    # 2. Truncar la fecha al primer día del mes (Agrupación temporal)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # 3. Sumar la demanda mensual por producto (Agrupación dimensional)
    df_mensual = (
        df.groupby(["product_id", "date"], as_index=False)["demand_real"]
        .sum()
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )

    if df_mensual.empty:
        raise ValueError("No hay datos válidos después de convertir la información a meses.")

    return df_mensual

def generar_demanda_sintetica(n_productos: int = 5, meses: int = 36, seed: int = 42) -> pd.DataFrame:
    """Genera demanda mensual sintética para pruebas."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(start="2023-01-01", periods=meses, freq="MS")
    dataframes = []

    for i in range(1, n_productos + 1):
        producto = f"PROD_{i:03d}"
        base = rng.integers(500, 2500)
        tendencia = rng.uniform(-10, 30)
        estacionalidad = rng.uniform(100, 400)
        ruido = rng.normal(0, base * 0.15, meses)
        tiempo = np.arange(meses)

        demanda = base + tendencia * tiempo + estacionalidad * np.sin(2 * np.pi * tiempo / 12) + ruido
        demanda = np.maximum(0, np.round(demanda)).astype(int)

        if i % 4 == 0:
            mascara_intermitente = rng.random(meses) < 0.45
            demanda = np.where(mascara_intermitente, 0, demanda)

        dataframes.append(
            pd.DataFrame(
                {
                    "date": fechas,
                    "product_id": producto,
                    "demand_real": demanda,
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)
