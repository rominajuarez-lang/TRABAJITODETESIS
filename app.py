import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# IMPORTACIÓN DE MÓDULOS LOCALES
# =========================================================
from datos import generar_demanda_sintetica, convertir_a_mensual
from generar_pronosticos import METODOS_PRONOSTICO, generar_forecast, generar_forecast_mejor_por_producto
from simulacion_inventario import (
    ParametrosInventario,
    simular_producto,
    calcular_kpis,
    optimizar_stock_seguridad,
    obtener_parametros_producto,
)
from visualizacion import grafico_forecast, grafico_inventario, grafico_tradeoff, formatear_comparacion
from tvu import preparar_tvu, resumen_tvu, grafico_cantidad_riesgo, grafico_valor_riesgo, formatear_tvu

warnings.filterwarnings("ignore")

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
# FUNCIONES AUXILIARES PARA DASHBOARD EJECUTIVO
# =========================================================
def normalizar_id(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip()


def leer_forecast_comercial_opcional(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Lee una hoja opcional Forecast_Comercial.
    Estructura esperada: date, product_id, forecast_company.
    Si no existe, devuelve DataFrame vacío para no romper la app.
    """
    nombres = {str(s).strip().lower(): s for s in xls.sheet_names}
    posibles = ["forecast_comercial", "forecast comercial", "pronostico_comercial", "pronóstico_comercial"]
    hoja = None
    for nombre in posibles:
        if nombre in nombres:
            hoja = nombres[nombre]
            break

    if hoja is None:
        return pd.DataFrame()

    df = pd.read_excel(xls, sheet_name=hoja)
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "producto": "product_id",
        "sku": "product_id",
        "grupo de demanda": "product_id",
        "id_producto": "product_id",
        "forecast": "forecast_company",
        "forecast comercial": "forecast_company",
        "pronostico": "forecast_company",
        "pronóstico": "forecast_company",
        "pronostico empresa": "forecast_company",
        "pronóstico empresa": "forecast_company",
        "forecast_empresa": "forecast_company",
        "forecast company": "forecast_company",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    requeridas = ["date", "product_id", "forecast_company"]
    if any(c not in df.columns for c in requeridas):
        return pd.DataFrame()

    df = df[requeridas].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = normalizar_id(df["product_id"])
    df["forecast_company"] = pd.to_numeric(df["forecast_company"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df = (
        df.groupby(["product_id", "date"], as_index=False)["forecast_company"]
        .sum()
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )
    return df


def calcular_resumen_forecast_2025(df_forecast_auto: pd.DataFrame, df_forecast_empresa: pd.DataFrame, df_tvu: pd.DataFrame):
    """
    Calcula el ahorro económico del forecast propuesto vs forecast comercial 2025.
    Ahorro = error valorizado empresa - error valorizado propuesta.
    """
    if df_forecast_empresa is None or df_forecast_empresa.empty:
        return pd.DataFrame(), {
            "ahorro_forecast": 0.0,
            "reduccion_error": 0.0,
            "error_empresa": 0.0,
            "error_propuesta": 0.0,
            "skus_mejorados": 0,
        }

    df_prop = df_forecast_auto[
        (df_forecast_auto["tipo_periodo"] == "Histórico")
        & (pd.to_datetime(df_forecast_auto["date"]).dt.year == 2025)
    ].copy()

    if df_prop.empty:
        return pd.DataFrame(), {
            "ahorro_forecast": 0.0,
            "reduccion_error": 0.0,
            "error_empresa": 0.0,
            "error_propuesta": 0.0,
            "skus_mejorados": 0,
        }

    costos = pd.DataFrame()
    if df_tvu is not None and not df_tvu.empty and {"product_id", "unit_value"}.issubset(df_tvu.columns):
        costos = df_tvu[["product_id", "unit_value"]].drop_duplicates("product_id").copy()
    else:
        costos = pd.DataFrame({"product_id": df_prop["product_id"].unique(), "unit_value": 1.0})

    df_emp = df_forecast_empresa[pd.to_datetime(df_forecast_empresa["date"]).dt.year == 2025].copy()

    df = df_prop.merge(df_emp, on=["product_id", "date"], how="inner")
    if df.empty:
        return pd.DataFrame(), {
            "ahorro_forecast": 0.0,
            "reduccion_error": 0.0,
            "error_empresa": 0.0,
            "error_propuesta": 0.0,
            "skus_mejorados": 0,
        }

    df = df.merge(costos, on="product_id", how="left")
    df["unit_value"] = pd.to_numeric(df["unit_value"], errors="coerce").fillna(0)

    filas = []
    for producto, sub in df.groupby("product_id"):
        real = sub["demand_real"].astype(float).to_numpy()
        empresa = sub["forecast_company"].astype(float).to_numpy()
        propuesta = sub["demand_forecast"].astype(float).to_numpy()
        costo = sub["unit_value"].astype(float).to_numpy()

        error_empresa = float(np.sum(np.abs(empresa - real) * costo))
        error_propuesta = float(np.sum(np.abs(propuesta - real) * costo))
        ahorro = error_empresa - error_propuesta

        filas.append({
            "Producto": producto,
            "Método": sub["method_used"].iloc[0] if "method_used" in sub.columns else "",
            "Error empresa S/": error_empresa,
            "Error propuesta S/": error_propuesta,
            "Ahorro potencial S/": ahorro,
        })

    resumen = pd.DataFrame(filas)
    error_empresa_total = resumen["Error empresa S/"].sum()
    error_propuesta_total = resumen["Error propuesta S/"].sum()
    ahorro_total = resumen["Ahorro potencial S/"].sum()
    reduccion = ((error_empresa_total - error_propuesta_total) / error_empresa_total * 100) if error_empresa_total > 0 else 0

    kpis = {
        "ahorro_forecast": float(ahorro_total),
        "reduccion_error": float(reduccion),
        "error_empresa": float(error_empresa_total),
        "error_propuesta": float(error_propuesta_total),
        "skus_mejorados": int((resumen["Ahorro potencial S/"] > 0).sum()),
    }
    return resumen, kpis


def preparar_resumen_modelos(df_comparacion: pd.DataFrame) -> pd.DataFrame:
    if df_comparacion is None or df_comparacion.empty or "Es mejor" not in df_comparacion.columns:
        return pd.DataFrame(columns=["Método", "Cantidad de Productos", "Porcentaje"])

    resumen_mejores = df_comparacion[df_comparacion["Es mejor"]].copy()
    if resumen_mejores.empty:
        return pd.DataFrame(columns=["Método", "Cantidad de Productos", "Porcentaje"])

    conteo = resumen_mejores["Método"].value_counts().reset_index()
    conteo.columns = ["Método", "Cantidad de Productos"]
    conteo["Porcentaje"] = conteo["Cantidad de Productos"] / conteo["Cantidad de Productos"].sum() * 100
    return conteo


def mostrar_dashboard_tvu(df_tvu, resumen_vencimientos, kpis_tvu):
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
        return

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("🔴 SKUs riesgo alto", f"{kpis_tvu['sku_alto']:,}")
    col_t2.metric("🟡 SKUs riesgo medio", f"{kpis_tvu['sku_medio']:,}")
    col_t3.metric("Stock en riesgo", f"{kpis_tvu['stock_riesgo']:,.0f}")
    col_t4.metric("Valor en riesgo", f"S/ {kpis_tvu['valor_riesgo']:,.2f}")

    st.info(f"SKU más crítico: **{kpis_tvu['sku_critico']}**")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(grafico_cantidad_riesgo(resumen_vencimientos), use_container_width=True)
    with col_g2:
        st.plotly_chart(grafico_valor_riesgo(resumen_vencimientos), use_container_width=True)

    st.markdown("### 🚨 Top 10 productos más críticos")
    top_10 = df_tvu[df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])].head(10)
    if top_10.empty:
        st.success("No hay productos en riesgo alto o medio.")
    else:
        st.dataframe(formatear_tvu(top_10), use_container_width=True, hide_index=True)

    st.markdown("### 📋 Detalle completo TVU")
    st.dataframe(formatear_tvu(df_tvu), use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 Descargar TVU (CSV)",
        data=df_tvu.to_csv(index=False).encode("utf-8"),
        file_name="reporte_tvu_vencimientos.csv",
        mime="text/csv",
        use_container_width=True,
    )


def mostrar_dashboard_ejecutivo(df_tvu, resumen_vencimientos, kpis_tvu, df_comparacion, resumen_forecast, kpis_forecast):
    st.title("📊 Dashboard Ejecutivo")
    st.caption("Vista general del portafolio: pronósticos, riesgo por vencimiento y oportunidad económica.")

    total_skus = int(df_tvu["product_id"].nunique()) if not df_tvu.empty and "product_id" in df_tvu.columns else 0
    valor_tvu_riesgo = float(kpis_tvu.get("valor_riesgo", 0))
    stock_riesgo = float(kpis_tvu.get("stock_riesgo", 0))
    ahorro_forecast = float(kpis_forecast.get("ahorro_forecast", 0))
    reduccion_error = float(kpis_forecast.get("reduccion_error", 0))
    impacto_identificado = valor_tvu_riesgo + ahorro_forecast

    conteo_modelos = preparar_resumen_modelos(df_comparacion)
    modelo_mas_usado = "Sin datos"
    if not conteo_modelos.empty:
        modelo_mas_usado = conteo_modelos.iloc[0]["Método"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SKU evaluados", f"{total_skus:,}")
    c2.metric("Ahorro forecast", f"S/ {ahorro_forecast:,.0f}")
    c3.metric("Valor TVU en riesgo", f"S/ {valor_tvu_riesgo:,.0f}")
    c4.metric("Impacto identificado", f"S/ {impacto_identificado:,.0f}")
    c5.metric("Modelo más usado", str(modelo_mas_usado))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💰 Top 10 ahorro potencial por forecast")
        if resumen_forecast is None or resumen_forecast.empty:
            st.info("No hay información de forecast comercial para calcular ahorro económico.")
        else:
            top_ahorro = resumen_forecast[resumen_forecast["Ahorro potencial S/"] > 0].copy()
            top_ahorro = top_ahorro.sort_values("Ahorro potencial S/", ascending=False).head(10)
            if top_ahorro.empty:
                st.info("No se detectaron SKUs donde la propuesta reduzca el error valorizado frente al forecast comercial.")
            else:
                fig_ahorro = px.bar(
                    top_ahorro,
                    x="Ahorro potencial S/",
                    y="Producto",
                    orientation="h",
                    title="SKUs con mayor ahorro potencial",
                    labels={"Ahorro potencial S/": "Ahorro potencial (S/)", "Producto": "SKU"},
                )
                fig_ahorro.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_ahorro, use_container_width=True)

    with col_b:
        st.subheader("⚠️ Valor en riesgo por vencimiento")
        if df_tvu.empty:
            st.info("No hay información TVU disponible.")
        else:
            resumen_riesgo = resumen_vencimientos[resumen_vencimientos["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])].copy()
            if resumen_riesgo.empty:
                st.success("No hay valor económico en riesgo alto o medio.")
            else:
                fig_riesgo = px.pie(
                    resumen_riesgo,
                    names="riesgo_tvu",
                    values="valor_en_riesgo",
                    hole=0.45,
                    title="Solo riesgo alto y medio",
                )
                fig_riesgo.update_traces(textposition="inside", textinfo="percent+label")
                fig_riesgo.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_riesgo, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🏆 Distribución de modelos ganadores")
        if conteo_modelos.empty:
            st.info("No hay comparación de modelos disponible.")
        else:
            fig_modelos = px.pie(
                conteo_modelos,
                names="Método",
                values="Cantidad de Productos",
                hole=0.45,
                title="Cantidad de SKUs por modelo seleccionado",
            )
            fig_modelos.update_traces(textposition="inside", textinfo="percent+label")
            fig_modelos.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_modelos, use_container_width=True)

    with col_d:
        st.subheader("📌 Lectura ejecutiva")
        st.metric("SKUs en riesgo alto/medio", f"{int(kpis_tvu.get('sku_alto', 0) + kpis_tvu.get('sku_medio', 0)):,}")
        st.metric("Stock en riesgo", f"{stock_riesgo:,.0f}")
        st.metric("Reducción del error valorizado", f"{reduccion_error:.1f}%")
        st.caption("El impacto identificado combina oportunidad de ahorro por forecast y valor económico en riesgo por vencimiento.")


# =========================================================
# SIDEBAR - CARGA DE DATOS ÚNICA
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

st.sidebar.header("1. Carga de datos")
modo_datos = st.sidebar.radio("Modo de datos", ["Generar datos sintéticos", "Subir Excel (Pestañas: Demanda y Datos)"])

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)
    df_real = generar_demanda_sintetica(n_productos=n_productos, meses=meses, seed=seed)
    df_parametros = pd.DataFrame()
    df_forecast_empresa = pd.DataFrame()
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo Excel unificado", type=["xlsx", "xls"])
    if archivo is None:
        st.info(
            "Sube un archivo Excel que contenga al menos dos pestañas:\n"
            "1. 'Demanda': historial de ventas (date, product_id, demand_real).\n"
            "2. 'Datos': maestro de artículos (GRUPO DE DEMANDA, initial_stock, tvu_months, unit_value, etc.).\n"
            "Opcional: 'Forecast_Comercial' para calcular ahorro económico del forecast."
        )
        st.stop()

    try:
        xls = pd.ExcelFile(archivo)

        if "Demanda" in xls.sheet_names:
            df_demanda_raw = pd.read_excel(xls, sheet_name="Demanda")
        else:
            df_demanda_raw = pd.read_excel(xls, sheet_name=0)

        df_demanda_raw.columns = [str(c).strip().lower() for c in df_demanda_raw.columns]
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
            "grupo de demanda": "product_id",
            "demanda": "demand_real",
            "venta": "demand_real",
            "ventas": "demand_real",
            "cantidad": "demand_real",
            "unidades": "demand_real",
        }
        df_demanda_raw = df_demanda_raw.rename(columns={c: alias.get(c, c) for c in df_demanda_raw.columns})
        df_real = convertir_a_mensual(df_demanda_raw)

        if "Datos" in xls.sheet_names:
            df_parametros = pd.read_excel(xls, sheet_name="Datos")
        else:
            st.error("⚠️ El archivo Excel no tiene una pestaña llamada 'Datos'. Por favor, agrégala y vuelve a subir el archivo.")
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

if modulo == "⚠️ TVU - Productos próximos a vencer":
    mostrar_dashboard_tvu(df_tvu, resumen_vencimientos, kpis_tvu)
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
    df_real, fecha_fin_pronostico=fecha_fin_pronostico
)

resumen_forecast, kpis_forecast = calcular_resumen_forecast_2025(df_forecast_auto, df_forecast_empresa, df_tvu)

if modulo == "📊 Vista General Ejecutiva":
    mostrar_dashboard_ejecutivo(df_tvu, resumen_vencimientos, kpis_tvu, df_comparacion, resumen_forecast, kpis_forecast)
    st.stop()

if modo_pronostico == "Manual: elegir un método":
    metodo_manual = st.sidebar.selectbox("Método manual", METODOS_PRONOSTICO)
    df_forecast = generar_forecast(df_real, metodo_manual, fecha_fin_pronostico=fecha_fin_pronostico)
else:
    metodo_manual = None
    df_forecast = df_forecast_auto

productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

sub_comparacion_producto = df_comparacion[df_comparacion["Producto"] == producto_sel].copy()
mejor_metodo_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "Método"].iloc[0]
mejor_wmape_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "wMAPE"].iloc[0]

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
sub_opt = optimizar_stock_seguridad(sub_forecast, politica, parametros_del_producto, ss_max=ss_max)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método usado", metodo_usado)
col2.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col3.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col4.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col5.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Mejor método",
    "📊 Datos y pronóstico",
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
    resumen_mejores = resumen_mejores[["Producto", "Método", "wMAPE", "Bias", "MAE"]].rename(
        columns={"Método": "Mejor método"}
    )

    col_graf, col_tabla = st.columns([1.2, 1])

    conteo_metodos = resumen_mejores["Mejor método"].value_counts().reset_index()
    conteo_metodos.columns = ["Método", "Cantidad de Productos"]
    conteo_metodos["Porcentaje"] = (conteo_metodos["Cantidad de Productos"] / len(resumen_mejores)) * 100

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
                "Porcentaje": st.column_config.NumberColumn("% del Portafolio", format="%.1f %%"),
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

    df_mostrar = resumen_mejores[resumen_mejores["Mejor método"].isin(filtro_metodos)].copy()
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
            "MAE": st.column_config.NumberColumn("MAE (Unidades)", format="%.2f"),
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
    st.write(f"Visualización del comportamiento histórico frente al modelo seleccionado: **{metodo_usado}**.")

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
        return ["background-color: #d4edda" if "✅" in str(val) else "" for val in row]

    st.dataframe(df_comp.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📦 Simulación Dinámica de Inventario")
    st.write("Evolución del stock físico frente a la demanda y generación de órdenes de compra según la política seleccionada.")

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

with tab4:
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
        "ss_months", "fill_rate", "lost_sales_units", "holding_cost", "stockout_cost", "total_cost"
    ]]
    df_sensibilidad.columns = [
        "Meses SS", "Fill Rate", "Ventas Perdidas (Unds)",
        "Costo Mantener (S/)", "Costo Quiebre (S/)", "Costo Total (S/)",
    ]

    def highlight_optimo(row):
        is_optimo = row["Meses SS"] == int(mejor["ss_months"])
        return ["background-color: #d4edda; font-weight: bold" if is_optimo else "" for _ in row]

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

with tab5:
    st.subheader("📋 Tablas de Datos y Reportes")
    st.write("Registros detallados de las proyecciones y simulaciones, formateados para exportación y análisis externo.")

    st.markdown("#### 📅 Datos Históricos y Pronóstico Futuro")
    df_fore_disp = sub_forecast.copy()
    df_fore_disp["date"] = pd.to_datetime(df_fore_disp["date"]).dt.strftime("%b %Y").str.upper()
    df_fore_disp["demand_real"] = df_fore_disp["demand_real"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
    df_fore_disp["demand_forecast"] = df_fore_disp["demand_forecast"].apply(lambda x: f"{x:,.0f}")
    df_fore_disp["method_wmape"] = df_fore_disp["method_wmape"].apply(lambda x: f"{x:.2%}")
    df_fore_disp["method_bias"] = df_fore_disp["method_bias"].apply(lambda x: f"{x:.2%}")
    df_fore_disp.columns = [
        "Fecha", "Producto", "Demanda Real", "Pronóstico",
        "Método Usado", "wMAPE", "Bias", "Tipo de Período",
    ]
    st.dataframe(df_fore_disp, use_container_width=True, hide_index=True)

    st.markdown("#### 📦 Registro Mensual de la Simulación de Inventario")
    df_sim_disp = sub_sim.copy()
    df_sim_disp["date"] = pd.to_datetime(df_sim_disp["date"]).dt.strftime("%b %Y").str.upper()
    df_sim_disp = df_sim_disp[[
        "date", "demand_real", "demand_forecast", "inventory_level",
        "order_placed", "arrivals", "sales_lost", "reorder_point_s",
    ]]
    df_sim_disp.columns = [
        "Mes", "Demanda Real", "Pronóstico", "Inventario Final",
        "Pedido Generado", "Llegadas (Recepción)", "Ventas Perdidas", "Punto Reorden (s)",
    ]
    for col in df_sim_disp.columns[1:]:
        df_sim_disp[col] = df_sim_disp[col].apply(lambda x: f"{x:,.0f}")
    st.dataframe(df_sim_disp, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Resultados de la Optimización de Stock de Seguridad")
    df_opt_disp = sub_opt.copy()
    df_opt_disp.columns = [
        "Meses SS", "Fill Rate", "Inv. Promedio", "Ventas Perdidas", "Meses Quiebre",
        "Total Órdenes", "Costo Órdenes (S/)", "Costo Almacenaje (S/)", "Costo Quiebre (S/)", "Costo Total (S/)",
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
