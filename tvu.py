import numpy as np
import pandas as pd
import plotly.express as px


def clasificar_riesgo_tvu(meses):
    if pd.isna(meses):
        return "Sin dato"
    if meses <= 3:
        return "🔴 Alto"
    elif meses <= 6:
        return "🟡 Medio"
    else:
        return "🟢 Bajo"


def preparar_tvu(df_parametros: pd.DataFrame) -> pd.DataFrame:
    df = df_parametros.copy()

    if df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "grupo de demanda": "product_id",
        "sku": "product_id",
        "producto": "product_id",
        "stock": "initial_stock",
        "stock_actual": "initial_stock",
        "stock actual": "initial_stock",
        "meses_vencer": "tvu_months",
        "meses vencimiento": "tvu_months",
        "meses_para_vencer": "tvu_months",
        "meses para vencer": "tvu_months",
        "costo_unitario": "unit_value",
        "costo unitario": "unit_value",
        "unit_cost": "unit_value",
        "valor_unitario": "unit_value",
    }

    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    requeridas = ["product_id", "initial_stock", "tvu_months", "unit_value"]
    faltantes = [c for c in requeridas if c not in df.columns]

    if faltantes:
        return pd.DataFrame()

    df = df[requeridas].copy()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["initial_stock"] = pd.to_numeric(df["initial_stock"], errors="coerce").fillna(0)
    df["tvu_months"] = pd.to_numeric(df["tvu_months"], errors="coerce")
    df["unit_value"] = pd.to_numeric(df["unit_value"], errors="coerce").fillna(0)

    df["valor_en_riesgo"] = df["initial_stock"] * df["unit_value"]
    df["riesgo_tvu"] = df["tvu_months"].apply(clasificar_riesgo_tvu)

    orden = {
        "🔴 Alto": 1,
        "🟡 Medio": 2,
        "🟢 Bajo": 3,
        "Sin dato": 4,
    }

    df["orden_riesgo"] = df["riesgo_tvu"].map(orden).fillna(9)

    df = df.sort_values(
        ["orden_riesgo", "tvu_months", "valor_en_riesgo"],
        ascending=[True, True, False]
    )

    return df


def resumen_tvu(df_tvu: pd.DataFrame):
    if df_tvu.empty:
        return pd.DataFrame(), {
            "sku_alto": 0,
            "sku_medio": 0,
            "sku_bajo": 0,
            "stock_riesgo": 0,
            "valor_riesgo": 0,
            "sku_critico": "Sin datos",
        }

    resumen = (
        df_tvu.groupby("riesgo_tvu", as_index=False)
        .agg(
            cantidad_sku=("product_id", "nunique"),
            stock_total=("initial_stock", "sum"),
            valor_en_riesgo=("valor_en_riesgo", "sum")
        )
    )

    alto_medio = df_tvu[df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])].copy()

    sku_critico = "Sin datos"
    if not df_tvu.empty:
        sku_critico = df_tvu.iloc[0]["product_id"]

    kpis = {
        "sku_alto": int((df_tvu["riesgo_tvu"] == "🔴 Alto").sum()),
        "sku_medio": int((df_tvu["riesgo_tvu"] == "🟡 Medio").sum()),
        "sku_bajo": int((df_tvu["riesgo_tvu"] == "🟢 Bajo").sum()),
        "stock_riesgo": float(alto_medio["initial_stock"].sum()) if not alto_medio.empty else 0,
        "valor_riesgo": float(alto_medio["valor_en_riesgo"].sum()) if not alto_medio.empty else 0,
        "sku_critico": sku_critico,
    }

    return resumen, kpis


def grafico_cantidad_riesgo(resumen: pd.DataFrame):
    fig = px.bar(
        resumen,
        x="riesgo_tvu",
        y="cantidad_sku",
        text="cantidad_sku",
        title="Cantidad de SKUs por nivel de riesgo de vencimiento",
        labels={
            "riesgo_tvu": "Nivel de riesgo",
            "cantidad_sku": "Cantidad de SKUs"
        }
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def grafico_valor_riesgo(resumen: pd.DataFrame):
    fig = px.pie(
        resumen,
        names="riesgo_tvu",
        values="valor_en_riesgo",
        hole=0.45,
        title="Valor económico comprometido por riesgo"
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def formatear_tvu(df_tvu: pd.DataFrame) -> pd.DataFrame:
    df = df_tvu.copy()

    if df.empty:
        return df

    df = df[[
        "product_id",
        "tvu_months",
        "initial_stock",
        "unit_value",
        "valor_en_riesgo",
        "riesgo_tvu"
    ]]

    df = df.rename(columns={
        "product_id": "Producto",
        "tvu_months": "Meses para vencer",
        "initial_stock": "Stock actual",
        "unit_value": "Valor unitario",
        "valor_en_riesgo": "Valor en riesgo",
        "riesgo_tvu": "Riesgo"
    })

    return df
