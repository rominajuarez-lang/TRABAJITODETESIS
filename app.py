import warnings
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# IMPORTACIÓN DE MÓDULOS LOCALES
# =========================================================
from datos import generar_demanda_sintetica, convertir_a_mensual
from generar_pronosticos import (
    METODOS_PRONOSTICO,
    generar_forecast,
    generar_forecast_mejor_por_producto,
)
from simulacion_inventario import (
    simular_producto,
    calcular_kpis,
    optimizar_stock_seguridad,
    obtener_parametros_producto,
)
from visualizacion import (
    grafico_forecast,
    grafico_inventario,
    grafico_tradeoff,
    formatear_comparacion,
)
from tvu import (
    preparar_tvu,
    resumen_tvu,
    grafico_cantidad_riesgo,
    grafico_valor_riesgo,
    formatear_tvu,
)

warnings.filterwarnings("ignore")


# =========================================================
# FUNCIONES AUXILIARES - FORECAST COMERCIAL Y DASHBOARD
# =========================================================
def normalizar_product_id(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def leer_forecast_comercial_opcional(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Lee el forecast comercial si existe.
    Prioriza la hoja Forecast_Comercial, pero si no existe busca automáticamente
    cualquier hoja que tenga date, product_id y forecast_company o sus alias.
    """
    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "día": "date",
        "dia": "date",
        "producto": "product_id",
        "sku": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "id_producto": "product_id",
        "grupo de demanda": "product_id",
        "forecast": "forecast_company",
        "forecast comercial": "forecast_company",
        "forecast_comercial": "forecast_company",
        "pronostico": "forecast_company",
        "pronóstico": "forecast_company",
        "pronostico_empresa": "forecast_company",
        "forecast empresa": "forecast_company",
        "forecast_company": "forecast_company",
    }

    hojas_a_revisar = []
    if "Forecast_Comercial" in xls.sheet_names:
        hojas_a_revisar.append("Forecast_Comercial")
    hojas_a_revisar += [h for h in xls.sheet_names if h not in hojas_a_revisar]

    requeridas = ["date", "product_id", "forecast_company"]

    for hoja in hojas_a_revisar:
        try:
            df = pd.read_excel(xls, sheet_name=hoja)
        except Exception:
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

        if not all(c in df.columns for c in requeridas):
            continue

        df = df[requeridas].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["product_id"] = normalizar_product_id(df["product_id"])
        df["forecast_company"] = pd.to_numeric(df["forecast_company"], errors="coerce").fillna(0)
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

        df = (
            df.groupby(["product_id", "date"], as_index=False)["forecast_company"]
            .sum()
            .sort_values(["product_id", "date"])
            .reset_index(drop=True)
        )

        if not df.empty:
            return df

    return pd.DataFrame(columns=requeridas)


def obtener_costos_unitarios(df_parametros: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae costo unitario desde la hoja Datos.
    Usa unit_value si existe; si no, intenta unit_cost o costo_unitario.
    """
    if df_parametros is None or df_parametros.empty:
        return pd.DataFrame(columns=["product_id", "unit_cost"])

    df = df_parametros.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "grupo de demanda": "product_id",
        "sku": "product_id",
        "producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "unit_value": "unit_cost",
        "unit cost": "unit_cost",
        "unit_cost": "unit_cost",
        "costo_unitario": "unit_cost",
        "costo unitario": "unit_cost",
        "valor_unitario": "unit_cost",
        "valor unitario": "unit_cost",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    if "product_id" not in df.columns:
        return pd.DataFrame(columns=["product_id", "unit_cost"])

    if "unit_cost" not in df.columns:
        df["unit_cost"] = 0

    out = df[["product_id", "unit_cost"]].copy()
    out["product_id"] = normalizar_product_id(out["product_id"])
    out["unit_cost"] = pd.to_numeric(out["unit_cost"], errors="coerce").fillna(0)
    out = out.drop_duplicates("product_id")
    return out


def calcular_ahorro_forecast_2025(
    df_forecast_auto: pd.DataFrame,
    df_forecast_empresa: pd.DataFrame,
    df_parametros: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula el ahorro potencial comparando:
    ventas reales 2025 vs forecast empresa 2025 vs forecast propuesto 2025.
    """
    kpis_cero = {
        "ahorro_total": 0.0,
        "error_empresa": 0.0,
        "error_propuesta": 0.0,
        "reduccion_error": 0.0,
        "skus_comparados": 0,
    }

    if (
        df_forecast_auto is None
        or df_forecast_auto.empty
        or df_forecast_empresa is None
        or df_forecast_empresa.empty
    ):
        return pd.DataFrame(), kpis_cero

    prop = df_forecast_auto[
        (df_forecast_auto["tipo_periodo"] == "Histórico")
        & (pd.to_datetime(df_forecast_auto["date"]).dt.year == 2025)
    ].copy()

    emp = df_forecast_empresa[
        pd.to_datetime(df_forecast_empresa["date"]).dt.year == 2025
    ].copy()

    if prop.empty or emp.empty:
        return pd.DataFrame(), kpis_cero

    prop["product_id"] = normalizar_product_id(prop["product_id"])
    emp["product_id"] = normalizar_product_id(emp["product_id"])

    costos = obtener_costos_unitarios(df_parametros)

    df = prop.merge(emp, on=["product_id", "date"], how="inner")
    df = df.merge(costos, on="product_id", how="left")
    df["unit_cost"] = df["unit_cost"].fillna(0)

    if df.empty:
        return pd.DataFrame(), kpis_cero

    filas = []

    for producto, sub in df.groupby("product_id"):
        real = pd.to_numeric(sub["demand_real"], errors="coerce").fillna(0)
        empresa = pd.to_numeric(sub["forecast_company"], errors="coerce").fillna(0)
        propuesta = pd.to_numeric(sub["demand_forecast"], errors="coerce").fillna(0)
        costo = pd.to_numeric(sub["unit_cost"], errors="coerce").fillna(0)

        error_empresa_s = ((empresa - real).abs() * costo).sum()
        error_propuesta_s = ((propuesta - real).abs() * costo).sum()
        ahorro = error_empresa_s - error_propuesta_s

        suma_real = real.sum()
        wmape_empresa = ((empresa - real).abs().sum() / suma_real) if suma_real > 0 else 0
        wmape_propuesta = ((propuesta - real).abs().sum() / suma_real) if suma_real > 0 else 0
        bias_empresa = ((empresa - real).sum() / suma_real) if suma_real > 0 else 0
        bias_propuesta = ((propuesta - real).sum() / suma_real) if suma_real > 0 else 0

        metodo = sub["method_used"].iloc[0] if "method_used" in sub.columns else ""

        filas.append(
            {
                "Producto": producto,
                "Mejor método": metodo,
                "Error empresa S/": error_empresa_s,
                "Error propuesta S/": error_propuesta_s,
                "Ahorro potencial S/": ahorro,
                "wMAPE empresa": wmape_empresa,
                "wMAPE propuesta": wmape_propuesta,
                "Bias empresa": bias_empresa,
                "Bias propuesta": bias_propuesta,
            }
        )

    resumen = pd.DataFrame(filas)

    error_empresa_total = resumen["Error empresa S/"].sum()
    error_propuesta_total = resumen["Error propuesta S/"].sum()
    ahorro_total = resumen["Ahorro potencial S/"].sum()

    reduccion_error = (
        ((error_empresa_total - error_propuesta_total) / error_empresa_total) * 100
        if error_empresa_total > 0
        else 0
    )

    kpis = {
        "ahorro_total": float(ahorro_total),
        "error_empresa": float(error_empresa_total),
        "error_propuesta": float(error_propuesta_total),
        "reduccion_error": float(reduccion_error),
        "skus_comparados": int(resumen["Producto"].nunique()),
    }

    return resumen, kpis


def calcular_detalle_ahorro_mensual_2025(
    df_forecast_auto: pd.DataFrame,
    df_forecast_empresa: pd.DataFrame,
    df_parametros: pd.DataFrame,
) -> pd.DataFrame:
    """
    Devuelve el detalle mensual 2025 por SKU:
    ventas reales, forecast comercial, forecast propuesto, errores valorizados y ahorro.
    """
    if (
        df_forecast_auto is None
        or df_forecast_auto.empty
        or df_forecast_empresa is None
        or df_forecast_empresa.empty
    ):
        return pd.DataFrame()

    prop = df_forecast_auto[
        (df_forecast_auto["tipo_periodo"] == "Histórico")
        & (pd.to_datetime(df_forecast_auto["date"]).dt.year == 2025)
    ].copy()

    emp = df_forecast_empresa[
        pd.to_datetime(df_forecast_empresa["date"]).dt.year == 2025
    ].copy()

    if prop.empty or emp.empty:
        return pd.DataFrame()

    prop["product_id"] = normalizar_product_id(prop["product_id"])
    emp["product_id"] = normalizar_product_id(emp["product_id"])

    costos = obtener_costos_unitarios(df_parametros)

    df = prop.merge(emp, on=["product_id", "date"], how="inner")
    df = df.merge(costos, on="product_id", how="left")
    df["unit_cost"] = df["unit_cost"].fillna(0)

    if df.empty:
        return pd.DataFrame()

    df["demand_real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["forecast_company"] = pd.to_numeric(df["forecast_company"], errors="coerce").fillna(0)
    df["demand_forecast"] = pd.to_numeric(df["demand_forecast"], errors="coerce").fillna(0)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0)

    df["error_empresa_unidades"] = (df["forecast_company"] - df["demand_real"]).abs()
    df["error_propuesta_unidades"] = (df["demand_forecast"] - df["demand_real"]).abs()
    df["error_empresa_soles"] = df["error_empresa_unidades"] * df["unit_cost"]
    df["error_propuesta_soles"] = df["error_propuesta_unidades"] * df["unit_cost"]
    df["ahorro_potencial_soles"] = df["error_empresa_soles"] - df["error_propuesta_soles"]

    df["exceso_empresa"] = (df["forecast_company"] - df["demand_real"]).clip(lower=0)
    df["faltante_empresa"] = (df["demand_real"] - df["forecast_company"]).clip(lower=0)
    df["exceso_propuesta"] = (df["demand_forecast"] - df["demand_real"]).clip(lower=0)
    df["faltante_propuesta"] = (df["demand_real"] - df["demand_forecast"]).clip(lower=0)

    columnas = [
        "date",
        "product_id",
        "demand_real",
        "forecast_company",
        "demand_forecast",
        "unit_cost",
        "error_empresa_soles",
        "error_propuesta_soles",
        "ahorro_potencial_soles",
        "exceso_empresa",
        "faltante_empresa",
        "exceso_propuesta",
        "faltante_propuesta",
        "method_used",
    ]

    return df[[c for c in columnas if c in df.columns]].sort_values(["product_id", "date"])


def grafico_ahorro_forecast(df_ahorro: pd.DataFrame):
    top = df_ahorro[df_ahorro["Ahorro potencial S/"] > 0].copy()
    top = top.sort_values("Ahorro potencial S/", ascending=False).head(10)

    fig = px.bar(
        top,
        x="Ahorro potencial S/",
        y="Producto",
        orientation="h",
        title="Top 10 SKUs con mayor ahorro potencial por forecast",
        labels={
            "Ahorro potencial S/": "Ahorro potencial (S/)",
            "Producto": "SKU",
        },
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def grafico_comparacion_forecast_sku(detalle_sku: pd.DataFrame, producto: str):
    df = detalle_sku.sort_values("date").copy()
    df["Mes"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y")

    fig = px.line(
        df,
        x="Mes",
        y=["demand_real", "forecast_company", "demand_forecast"],
        markers=True,
        title=f"Ventas reales vs forecast comercial vs propuesta - {producto}",
        labels={"value": "Unidades", "Mes": "Mes", "variable": "Serie"},
    )
    fig.for_each_trace(lambda t: t.update(name={
        "demand_real": "Ventas reales",
        "forecast_company": "Forecast comercial",
        "demand_forecast": "Forecast propuesto",
    }.get(t.name, t.name)))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified")
    return fig


def grafico_ahorro_mensual_sku(detalle_sku: pd.DataFrame, producto: str):
    df = detalle_sku.sort_values("date").copy()
    df["Mes"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y")

    fig = px.bar(
        df,
        x="Mes",
        y="ahorro_potencial_soles",
        title=f"Ahorro potencial mensual - {producto}",
        labels={"ahorro_potencial_soles": "Ahorro potencial (S/)", "Mes": "Mes"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def grafico_tvu_alto_medio(resumen_vencimientos: pd.DataFrame):
    df = resumen_vencimientos.copy()
    df = df[df["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])]

    fig = px.pie(
        df,
        names="riesgo_tvu",
        values="valor_en_riesgo",
        hole=0.45,
        title="Valor en riesgo por vencimiento",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def grafico_modelos_ganadores(df_comparacion: pd.DataFrame):
    mejores = df_comparacion[df_comparacion["Es mejor"]].copy()
    conteo = mejores["Método"].value_counts().reset_index()
    conteo.columns = ["Método", "Cantidad de SKUs"]

    fig = px.pie(
        conteo,
        names="Método",
        values="Cantidad de SKUs",
        hole=0.45,
        title="Distribución de modelos ganadores",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Inventory Intelligence Framework",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Framework de Optimización de Inventarios")
st.caption(
    "Pronóstico mensual + selección automática del mejor método por producto + simulación + optimización de inventarios"
)

# =========================================================
# SIDEBAR - SELECCIÓN DE MÓDULO
# =========================================================
st.sidebar.header("Módulo")

modulo = st.sidebar.radio(
    "Seleccione una herramienta",
    [
        "📊 Vista General Ejecutiva",
        "📈 Pronósticos e Inventarios",
        "⚠️ TVU - Productos próximos a vencer",
    ],
)

# =========================================================
# SIDEBAR - CARGA DE DATOS ÚNICA
# =========================================================
st.sidebar.header("1. Carga de datos")

modo_datos = st.sidebar.radio(
    "Modo de datos",
    ["Generar datos sintéticos", "Subir Excel (Pestañas: Demanda y Datos)"],
)

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)

    df_real = generar_demanda_sintetica(
        n_productos=n_productos,
        meses=meses,
        seed=seed,
    )
    df_parametros = pd.DataFrame()
    df_forecast_empresa = pd.DataFrame(columns=["date", "product_id", "forecast_company"])

else:
    archivo = st.sidebar.file_uploader(
        "Sube tu archivo Excel unificado",
        type=["xlsx", "xls"],
    )

    if archivo is None:
        st.info(
            "Sube un archivo Excel que contenga al menos dos pestañas:\n"
            "1. 'Demanda': historial con date, product_id, demand_real.\n"
            "2. 'Datos': maestro de artículos.\n"
            "Opcional: 'Forecast_Comercial' con date, product_id, forecast_company."
        )
        st.stop()

    try:
        xls = pd.ExcelFile(archivo)

        if "Demanda" in xls.sheet_names:
            df_demanda_raw = pd.read_excel(xls, sheet_name="Demanda")
        else:
            df_demanda_raw = pd.read_excel(xls, sheet_name=0)

        df_demanda_raw.columns = [
            str(c).strip().lower() for c in df_demanda_raw.columns
        ]

        alias = {
            "fecha": "date",
            "mes": "date",
            "periodo": "date",
            "día": "date",
            "dia": "date",
            "producto": "product_id",
            "sku": "product_id",
            "id_producto": "product_id",
            "codigo": "product_id",
            "código": "product_id",
            "demanda": "demand_real",
            "venta": "demand_real",
            "ventas": "demand_real",
            "cantidad": "demand_real",
            "unidades": "demand_real",
        }

        df_demanda_raw = df_demanda_raw.rename(
            columns={c: alias.get(c, c) for c in df_demanda_raw.columns}
        )

        df_real = convertir_a_mensual(df_demanda_raw)

        if "Datos" in xls.sheet_names:
            df_parametros = pd.read_excel(xls, sheet_name="Datos")
        else:
            st.error(
                "⚠️ El archivo Excel no tiene una pestaña llamada 'Datos'. "
                "Por favor, agrégala y vuelve a subir el archivo."
            )
            st.stop()

        df_forecast_empresa = leer_forecast_comercial_opcional(xls)

    except Exception as e:
        st.error(f"Error procesando el archivo: {str(e)}")
        st.stop()

# =========================================================
# TVU - RIESGO DE VENCIMIENTO
# =========================================================
df_tvu = preparar_tvu(df_parametros)
resumen_vencimientos, kpis_tvu = resumen_tvu(df_tvu)

if df_tvu.empty:
    total_skus_tvu = 0
    valor_alto = 0.0
    valor_medio = 0.0
    valor_bajo = 0.0
    valor_tvu_riesgo = 0.0
else:
    total_skus_tvu = df_tvu["product_id"].nunique()
    valor_alto = df_tvu.loc[
        df_tvu["riesgo_tvu"] == "🔴 Alto",
        "valor_en_riesgo",
    ].sum()
    valor_medio = df_tvu.loc[
        df_tvu["riesgo_tvu"] == "🟡 Medio",
        "valor_en_riesgo",
    ].sum()
    valor_bajo = df_tvu.loc[
        df_tvu["riesgo_tvu"] == "🟢 Bajo",
        "valor_en_riesgo",
    ].sum()
    # Valor realmente comprometido: alto + medio. No incluye bajo.
    valor_tvu_riesgo = valor_alto + valor_medio

# =========================================================
# MÓDULO TVU INDEPENDIENTE
# =========================================================
if modulo == "⚠️ TVU - Productos próximos a vencer":
    st.subheader("⚠️ Infografía TVU: Productos próximos a vencer")

    st.write(
        "Clasificación de productos según los meses restantes para su vencimiento. "
        "Riesgo alto: hasta 3 meses; riesgo medio: más de 3 y hasta 6 meses; "
        "riesgo bajo: más de 6 meses."
    )

    if df_tvu.empty:
        st.warning(
            "No se pudo construir la infografía TVU. Verifica que la hoja 'Datos' tenga columnas como: "
            "GRUPO DE DEMANDA, initial_stock, tvu_months y unit_value."
        )
    else:
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)

        col_t1.metric("🔴 SKUs riesgo alto", f"{kpis_tvu['sku_alto']:,}")
        col_t2.metric("🟡 SKUs riesgo medio", f"{kpis_tvu['sku_medio']:,}")
        col_t3.metric("Stock en riesgo", f"{kpis_tvu['stock_riesgo']:,.0f}")
        col_t4.metric("Valor en riesgo", f"S/ {kpis_tvu['valor_riesgo']:,.2f}")

        st.info(f"SKU más crítico: **{kpis_tvu['sku_critico']}**")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.plotly_chart(
                grafico_cantidad_riesgo(resumen_vencimientos),
                use_container_width=True,
            )

        with col_g2:
            st.plotly_chart(
                grafico_valor_riesgo(resumen_vencimientos),
                use_container_width=True,
            )

        st.markdown("### 🚨 Top 10 productos más críticos")

        top_10 = df_tvu[
            df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])
        ].head(10)

        if top_10.empty:
            st.success("No hay productos en riesgo alto o medio.")
        else:
            st.dataframe(
                formatear_tvu(top_10),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 📋 Detalle completo TVU")

        st.dataframe(
            formatear_tvu(df_tvu),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="📥 Descargar TVU (CSV)",
            data=df_tvu.to_csv(index=False).encode("utf-8"),
            file_name="reporte_tvu_vencimientos.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.stop()

# =========================================================
# PRONÓSTICO MENSUAL
# =========================================================
st.sidebar.header("2. Pronóstico mensual")

modo_pronostico = st.sidebar.selectbox(
    "Selección del método",
    ["Automático: mejor método por producto", "Manual: elegir un método"],
)

ultima_fecha_historica = pd.to_datetime(df_real["date"].max()).to_period("M").to_timestamp()

fecha_fin_pronostico = st.sidebar.date_input(
    "Pronosticar hasta",
    value=pd.Timestamp("2026-12-01"),
    min_value=ultima_fecha_historica.date(),
)

fecha_fin_pronostico = pd.to_datetime(fecha_fin_pronostico).to_period("M").to_timestamp()

df_forecast_auto, df_comparacion = generar_forecast_mejor_por_producto(
    df_real,
    fecha_fin_pronostico=fecha_fin_pronostico,
)

# =========================================================
# RESUMEN FORECAST PARA VISTA GENERAL
# =========================================================
total_skus_forecast = df_real["product_id"].nunique()

df_ahorro_forecast, kpis_forecast = calcular_ahorro_forecast_2025(
    df_forecast_auto=df_forecast_auto,
    df_forecast_empresa=df_forecast_empresa,
    df_parametros=df_parametros,
)

df_detalle_ahorro_forecast = calcular_detalle_ahorro_mensual_2025(
    df_forecast_auto=df_forecast_auto,
    df_forecast_empresa=df_forecast_empresa,
    df_parametros=df_parametros,
)

ahorro_total = kpis_forecast["ahorro_total"]
reduccion_error = kpis_forecast["reduccion_error"]
skus_comparados_forecast = kpis_forecast["skus_comparados"]

resumen_mejores_exec = df_comparacion[df_comparacion["Es mejor"]].copy()
modelo_mas_usado = (
    resumen_mejores_exec["Método"].mode().iloc[0]
    if not resumen_mejores_exec.empty
    else "Sin datos"
)

# =========================================================
# MÓDULO VISTA GENERAL EJECUTIVA
# =========================================================
if modulo == "📊 Vista General Ejecutiva":
    st.title("📊 Dashboard Ejecutivo")
    st.caption("Vista general del desempeño del portafolio: forecast, riesgo de vencimiento y modelos ganadores.")

    total_skus = max(total_skus_tvu, total_skus_forecast)
    impacto_identificado = ahorro_total + valor_tvu_riesgo

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("SKU evaluados", f"{total_skus:,}")
    c2.metric("Ahorro potencial forecast", f"S/ {ahorro_total:,.0f}")
    c3.metric("Valor en riesgo TVU", f"S/ {valor_tvu_riesgo:,.0f}")
    c4.metric("Impacto identificado", f"S/ {impacto_identificado:,.0f}")
    c5.metric("Modelo más usado", modelo_mas_usado)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Ahorro potencial por forecast")

        if df_ahorro_forecast.empty or df_ahorro_forecast["Ahorro potencial S/"].max() <= 0:
            st.info(
                "No se calculó ahorro potencial. Para activarlo, el Excel debe incluir "
                "Forecast_Comercial con date, product_id y forecast_company, y la hoja Datos debe tener unit_value o unit_cost."
            )
        else:
            st.plotly_chart(
                grafico_ahorro_forecast(df_ahorro_forecast),
                use_container_width=True,
            )

    with col_b:
        st.subheader("⚠️ Valor en riesgo por vencimiento")

        if resumen_vencimientos.empty or valor_tvu_riesgo <= 0:
            st.info("No hay productos en riesgo alto o medio.")
        else:
            st.plotly_chart(
                grafico_tvu_alto_medio(resumen_vencimientos),
                use_container_width=True,
            )

    st.divider()

    st.subheader("🧠 Distribución de modelos ganadores")

    if resumen_mejores_exec.empty:
        st.info("No hay métodos ganadores disponibles.")
    else:
        st.plotly_chart(
            grafico_modelos_ganadores(df_comparacion),
            use_container_width=True,
        )

    st.stop()

# =========================================================
# CONFIGURACIÓN DEL MÓDULO PRONÓSTICOS E INVENTARIOS
# =========================================================
if modo_pronostico == "Manual: elegir un método":
    metodo_manual = st.sidebar.selectbox("Método manual", METODOS_PRONOSTICO)
    df_forecast = generar_forecast(
        df_real,
        metodo_manual,
        fecha_fin_pronostico=fecha_fin_pronostico,
    )
else:
    metodo_manual = None
    df_forecast = df_forecast_auto

productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

sub_comparacion_producto = df_comparacion[
    df_comparacion["Producto"] == producto_sel
].copy()

mejor_metodo_producto = sub_comparacion_producto.loc[
    sub_comparacion_producto["Es mejor"],
    "Método",
].iloc[0]

mejor_wmape_producto = sub_comparacion_producto.loc[
    sub_comparacion_producto["Es mejor"],
    "wMAPE",
].iloc[0]

if modo_pronostico == "Automático: mejor método por producto":
    st.sidebar.success(f"Método elegido para {producto_sel}: {mejor_metodo_producto}")
else:
    st.sidebar.info(f"Mejor método para {producto_sel}: {mejor_metodo_producto}")

# =========================================================
# POLÍTICA DE INVENTARIO
# =========================================================
st.sidebar.header("3. Política de Inventario")

politica = st.sidebar.selectbox(
    "Política (Modo Simulación)",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

ss_max = st.sidebar.slider("Máximo SS para optimizar (meses)", 1, 24, 6)

parametros_del_producto = obtener_parametros_producto(df_parametros, producto_sel)

# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
sub_forecast = df_forecast[df_forecast["product_id"] == producto_sel].copy()
metodo_usado = sub_forecast["method_used"].iloc[0]

sub_sim = simular_producto(sub_forecast, politica, parametros_del_producto)
kpis = calcular_kpis(sub_sim, parametros_del_producto)
sub_opt = optimizar_stock_seguridad(
    sub_forecast,
    politica,
    parametros_del_producto,
    ss_max=ss_max,
)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método usado", metodo_usado)
col2.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col3.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col4.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col5.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Mejor método",
    "📊 Datos y pronóstico",
    "💰 Comparación Forecast",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

with tab1:
    st.subheader("🏆 Análisis Estratégico: Mejor Método por Producto")
    st.write(
        "El framework evalúa todos los modelos mediante Validación Cruzada y selecciona el ganador "
        "basado en el menor wMAPE, utilizando el RMSE y el Bias como criterios de desempate."
    )

    resumen_mejores = (
        df_comparacion[df_comparacion["Es mejor"]]
        .copy()
        .sort_values("Producto")
    )

    resumen_mejores = resumen_mejores[[
        "Producto",
        "Método",
        "wMAPE",
        "Bias",
        "MAE",
    ]].rename(columns={"Método": "Mejor método"})

    col_graf, col_tabla = st.columns([1.2, 1])

    conteo_metodos = resumen_mejores["Mejor método"].value_counts().reset_index()
    conteo_metodos.columns = ["Método", "Cantidad de Productos"]
    conteo_metodos["Porcentaje"] = (
        conteo_metodos["Cantidad de Productos"] / len(resumen_mejores)
    ) * 100

    with col_graf:
        fig_donut = px.pie(
            conteo_metodos,
            names="Método",
            values="Cantidad de Productos",
            hole=0.45,
            title="Distribución de Métodos Ganadores",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_tabla:
        st.write("<br>", unsafe_allow_html=True)
        st.markdown("**Resumen de Asignación de Modelos**")
        st.dataframe(
            conteo_metodos,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Cantidad de Productos": st.column_config.ProgressColumn(
                    "Cantidad",
                    format="%d",
                    min_value=0,
                    max_value=int(conteo_metodos["Cantidad de Productos"].max()),
                ),
                "Porcentaje": st.column_config.NumberColumn(
                    "% del Portafolio",
                    format="%.1f %%",
                ),
            },
        )

    st.divider()

    st.subheader("🔎 Detalle por Producto")

    metodos_disponibles = conteo_metodos["Método"].tolist()
    filtro_metodos = st.multiselect(
        "Filtra la tabla por Método Ganador:",
        options=metodos_disponibles,
        default=metodos_disponibles,
    )

    df_mostrar = resumen_mejores[
        resumen_mejores["Mejor método"].isin(filtro_metodos)
    ].copy()

    df_mostrar["wMAPE"] = df_mostrar["wMAPE"] * 100
    df_mostrar["Bias"] = df_mostrar["Bias"] * 100

    st.dataframe(
        df_mostrar,
        hide_index=True,
        use_container_width=True,
        column_config={
            "wMAPE": st.column_config.NumberColumn(
                "wMAPE (%)",
                help="Error Porcentual Absoluto Medio Ponderado",
                format="%.2f %%",
            ),
            "Bias": st.column_config.NumberColumn(
                "Bias (%)",
                help="Sesgo del pronóstico (Positivo = Sobrepronóstico, Negativo = Subpronóstico)",
                format="%.2f %%",
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE (Unidades)",
                format="%.2f",
            ),
        },
    )

    st.write("<br>", unsafe_allow_html=True)

    csv_mejores = resumen_mejores.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Descargar detalle completo en CSV",
        data=csv_mejores,
        file_name="mejor_metodo_por_producto.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("📊 Análisis de Demanda y Proyección")
    st.write(
        f"Visualización del comportamiento histórico frente al modelo seleccionado: **{metodo_usado}**."
    )

    col_g1, col_g2 = st.columns([3, 1])

    with col_g1:
        fig = grafico_forecast(sub_forecast)
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        st.markdown("### 🎯 Resumen del Modelo")
        st.metric("Método Seleccionado", metodo_usado)
        st.metric("wMAPE (Error)", f"{mejor_wmape_producto:.2%}")

        st.markdown("---")
        st.markdown("**Insights clave:**")

        if mejor_wmape_producto < 0.20:
            st.success("Modelo de alta precisión. Apto para compras automáticas.")
        elif mejor_wmape_producto < 0.50:
            st.warning("Modelo con precisión moderada. Se recomienda revisión manual.")
        else:
            st.error("Precisión baja. Posible demanda errática o quiebre de stock.")

    st.markdown("### 📋 Comparativa de Métodos (Validación Cruzada)")

    df_comp = formatear_comparacion(sub_comparacion_producto)

    def highlight_best(row):
        return [
            "background-color: #d4edda" if "✅" in str(val) else ""
            for val in row
        ]

    st.dataframe(
        df_comp.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("💰 Comparación Forecast 2025")
    st.write(
        "Comparación mensual entre ventas reales, forecast comercial y forecast propuesto. "
        "El ahorro se calcula como la diferencia entre el error valorizado de la empresa y el error valorizado de la propuesta."
    )

    detalle_sku = pd.DataFrame()
    if not df_detalle_ahorro_forecast.empty:
        detalle_sku = df_detalle_ahorro_forecast[
            df_detalle_ahorro_forecast["product_id"] == normalizar_product_id(pd.Series([producto_sel])).iloc[0]
        ].copy()

    if df_detalle_ahorro_forecast.empty:
        st.warning(
            "No se pudo calcular la comparación económica. Verifica que exista forecast comercial con columnas "
            "date, product_id y forecast_company, y que la hoja Datos tenga unit_value o unit_cost."
        )
    elif detalle_sku.empty:
        st.warning("Este SKU no tiene coincidencias entre ventas reales 2025, forecast comercial y forecast propuesto.")
    else:
        ahorro_sku = detalle_sku["ahorro_potencial_soles"].sum()
        error_empresa_sku = detalle_sku["error_empresa_soles"].sum()
        error_propuesta_sku = detalle_sku["error_propuesta_soles"].sum()
        reduccion_sku = ((error_empresa_sku - error_propuesta_sku) / error_empresa_sku) if error_empresa_sku > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Error empresa 2025", f"S/ {error_empresa_sku:,.2f}")
        k2.metric("Error propuesta 2025", f"S/ {error_propuesta_sku:,.2f}")
        k3.metric("Ahorro potencial SKU", f"S/ {ahorro_sku:,.2f}")
        k4.metric("Reducción del error", f"{reduccion_sku:.2%}")

        col_cf1, col_cf2 = st.columns([1.4, 1])

        with col_cf1:
            st.plotly_chart(
                grafico_comparacion_forecast_sku(detalle_sku, producto_sel),
                use_container_width=True,
            )

        with col_cf2:
            st.plotly_chart(
                grafico_ahorro_mensual_sku(detalle_sku, producto_sel),
                use_container_width=True,
            )

        st.markdown("### 📋 Detalle mensual del SKU")
        detalle_mostrar = detalle_sku.copy()
        detalle_mostrar["date"] = pd.to_datetime(detalle_mostrar["date"]).dt.strftime("%b %Y").str.upper()
        detalle_mostrar = detalle_mostrar.rename(columns={
            "date": "Mes",
            "product_id": "Producto",
            "demand_real": "Ventas reales",
            "forecast_company": "Forecast comercial",
            "demand_forecast": "Forecast propuesto",
            "unit_cost": "Costo unitario",
            "error_empresa_soles": "Error empresa S/",
            "error_propuesta_soles": "Error propuesta S/",
            "ahorro_potencial_soles": "Ahorro potencial S/",
            "exceso_empresa": "Exceso empresa",
            "faltante_empresa": "Faltante empresa",
            "exceso_propuesta": "Exceso propuesta",
            "faltante_propuesta": "Faltante propuesta",
            "method_used": "Método usado",
        })

        columnas_mostrar = [
            "Mes", "Producto", "Ventas reales", "Forecast comercial", "Forecast propuesto",
            "Costo unitario", "Error empresa S/", "Error propuesta S/", "Ahorro potencial S/",
            "Exceso empresa", "Faltante empresa", "Exceso propuesta", "Faltante propuesta", "Método usado"
        ]
        columnas_mostrar = [c for c in columnas_mostrar if c in detalle_mostrar.columns]

        st.dataframe(
            detalle_mostrar[columnas_mostrar],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ventas reales": st.column_config.NumberColumn(format="%,.0f"),
                "Forecast comercial": st.column_config.NumberColumn(format="%,.0f"),
                "Forecast propuesto": st.column_config.NumberColumn(format="%,.0f"),
                "Costo unitario": st.column_config.NumberColumn(format="S/ %.2f"),
                "Error empresa S/": st.column_config.NumberColumn(format="S/ %.2f"),
                "Error propuesta S/": st.column_config.NumberColumn(format="S/ %.2f"),
                "Ahorro potencial S/": st.column_config.NumberColumn(format="S/ %.2f"),
                "Exceso empresa": st.column_config.NumberColumn(format="%,.0f"),
                "Faltante empresa": st.column_config.NumberColumn(format="%,.0f"),
                "Exceso propuesta": st.column_config.NumberColumn(format="%,.0f"),
                "Faltante propuesta": st.column_config.NumberColumn(format="%,.0f"),
            },
        )

        st.download_button(
            label="📥 Descargar comparación forecast del SKU (CSV)",
            data=detalle_sku.to_csv(index=False).encode("utf-8"),
            file_name=f"comparacion_forecast_2025_{producto_sel}.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tab4:
    st.subheader("📦 Simulación Dinámica de Inventario")
    st.write(
        "Evolución del stock físico frente a la demanda y generación de órdenes de compra según la política seleccionada."
    )

    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Indicadores de Desempeño (KPIs) del Escenario Actual")

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        st.markdown("**Nivel de Servicio**")
        fill_rate_val = kpis["fill_rate"]
        st.metric("Fill Rate", f"{fill_rate_val:.2%}")
        st.progress(min(fill_rate_val, 1.0))

        if fill_rate_val < 0.90:
            st.error(f"¡Atención! Ventas perdidas: {int(kpis['lost_sales_units'])} unds.")
        else:
            st.success("Nivel de servicio óptimo.")

    with col_kpi2:
        st.markdown("**Operaciones de Almacén**")
        st.metric("Inventario Promedio", f"{kpis['avg_inventory']:,.0f} unds")
        st.metric("Órdenes Emitidas", f"{kpis['orders']} pedidos")
        st.metric("Meses con Quiebre", f"{kpis['stockout_months']} meses")

    with col_kpi3:
        st.markdown("**Análisis Financiero**")
        st.metric("Costo de Mantener", f"S/ {kpis['holding_cost']:,.2f}")
        st.metric("Costo de Quiebre (Penalidad)", f"S/ {kpis['stockout_cost']:,.2f}")
        st.metric("Costo de Ordenar", f"S/ {kpis['ordering_cost']:,.2f}")

    st.info(f"**Costo Total de la Política Actual:** S/ {kpis['total_cost']:,.2f}")

with tab5:
    st.subheader("🎯 Optimización Financiera del Stock de Seguridad")
    st.write(
        "Análisis de sensibilidad (Trade-off) para encontrar el equilibrio exacto entre "
        "el costo de mantener inventario inmovilizado y la penalidad por ventas perdidas."
    )

    st.success(
        f"**Recomendación del Sistema:** Para el producto {producto_sel}, el stock de seguridad óptimo "
        f"es **{int(mejor['ss_months'])} meses**. \n\n"
        f"Esta configuración proyecta un Costo Total mínimo de **S/ {mejor['total_cost']:,.2f}** "
        f"alcanzando un Nivel de Servicio (Fill Rate) del **{mejor['fill_rate']:.2%}**."
    )

    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Tabla de Sensibilidad de Escenarios")

    df_sensibilidad = sub_opt.copy()

    df_sensibilidad = df_sensibilidad[[
        "ss_months",
        "fill_rate",
        "lost_sales_units",
        "holding_cost",
        "stockout_cost",
        "total_cost",
    ]]

    df_sensibilidad.columns = [
        "Meses SS",
        "Fill Rate",
        "Ventas Perdidas (Unds)",
        "Costo Mantener (S/)",
        "Costo Quiebre (S/)",
        "Costo Total (S/)",
    ]

    def highlight_optimo(row):
        is_optimo = row["Meses SS"] == int(mejor["ss_months"])
        return [
            "background-color: #d4edda; font-weight: bold" if is_optimo else ""
            for _ in row
        ]

    st.dataframe(
        df_sensibilidad.style.apply(highlight_optimo, axis=1).format({
            "Fill Rate": "{:.2%}",
            "Ventas Perdidas (Unds)": "{:,.0f}",
            "Costo Mantener (S/)": "{:,.2f}",
            "Costo Quiebre (S/)": "{:,.2f}",
            "Costo Total (S/)": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab6:
    st.subheader("📋 Tablas de Datos y Reportes")
    st.write(
        "Registros detallados de las proyecciones y simulaciones, formateados para exportación y análisis externo."
    )

    st.markdown("#### 📅 Datos Históricos y Pronóstico Futuro")

    df_fore_disp = sub_forecast.copy()

    df_fore_disp["date"] = pd.to_datetime(df_fore_disp["date"]).dt.strftime("%b %Y").str.upper()
    df_fore_disp["demand_real"] = df_fore_disp["demand_real"].apply(
        lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
    )
    df_fore_disp["demand_forecast"] = df_fore_disp["demand_forecast"].apply(
        lambda x: f"{x:,.0f}"
    )
    df_fore_disp["method_wmape"] = df_fore_disp["method_wmape"].apply(
        lambda x: f"{x:.2%}"
    )
    df_fore_disp["method_bias"] = df_fore_disp["method_bias"].apply(
        lambda x: f"{x:.2%}"
    )

    df_fore_disp.columns = [
        "Fecha",
        "Producto",
        "Demanda Real",
        "Pronóstico",
        "Método Usado",
        "wMAPE",
        "Bias",
        "Tipo de Período",
    ]

    st.dataframe(df_fore_disp, use_container_width=True, hide_index=True)

    st.markdown("#### 📦 Registro Mensual de la Simulación de Inventario")

    df_sim_disp = sub_sim.copy()
    df_sim_disp["date"] = pd.to_datetime(df_sim_disp["date"]).dt.strftime("%b %Y").str.upper()

    df_sim_disp = df_sim_disp[[
        "date",
        "demand_real",
        "demand_forecast",
        "inventory_level",
        "order_placed",
        "arrivals",
        "sales_lost",
        "reorder_point_s",
    ]]

    df_sim_disp.columns = [
        "Mes",
        "Demanda Real",
        "Pronóstico",
        "Inventario Final",
        "Pedido Generado",
        "Llegadas (Recepción)",
        "Ventas Perdidas",
        "Punto Reorden (s)",
    ]

    for col in df_sim_disp.columns[1:]:
        df_sim_disp[col] = df_sim_disp[col].apply(lambda x: f"{x:,.0f}")

    st.dataframe(df_sim_disp, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Resultados de la Optimización de Stock de Seguridad")

    df_opt_disp = sub_opt.copy()

    df_opt_disp.columns = [
        "Meses SS",
        "Fill Rate",
        "Inv. Promedio",
        "Ventas Perdidas",
        "Meses Quiebre",
        "Total Órdenes",
        "Costo Órdenes (S/)",
        "Costo Almacenaje (S/)",
        "Costo Quiebre (S/)",
        "Costo Total (S/)",
    ]

    st.dataframe(
        df_opt_disp.style.format({
            "Fill Rate": "{:.2%}",
            "Inv. Promedio": "{:,.0f}",
            "Ventas Perdidas": "{:,.0f}",
            "Meses Quiebre": "{:.0f}",
            "Total Órdenes": "{:.0f}",
            "Costo Órdenes (S/)": "{:,.2f}",
            "Costo Almacenaje (S/)": "{:,.2f}",
            "Costo Quiebre (S/)": "{:,.2f}",
            "Costo Total (S/)": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            label="📥 Descargar Pronóstico (CSV)",
            data=sub_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"pronostico_{producto_sel}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_d2:
        st.download_button(
            label="📥 Descargar Simulación (CSV)",
            data=sub_sim.to_csv(index=False).encode("utf-8"),
            file_name=f"simulacion_{producto_sel}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_d3:
        st.download_button(
            label="📥 Descargar Comparativa Métodos (CSV)",
            data=df_comparacion.to_csv(index=False).encode("utf-8"),
            file_name="comparacion_metodos_portafolio.csv",
            mime="text/csv",
            use_container_width=True,
        )
