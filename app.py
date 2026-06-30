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
# FUNCIONES AUXILIARES APP
# =========================================================
@st.cache_data(show_spinner="Calculando modelos de pronóstico por producto...")
def ejecutar_forecast_cache(df_real: pd.DataFrame, fecha_fin_pronostico: pd.Timestamp):
    return generar_forecast_mejor_por_producto(
        df_real,
        fecha_fin_pronostico=fecha_fin_pronostico,
    )


def construir_resumen_mejores(df_comparacion: pd.DataFrame) -> pd.DataFrame:
    if df_comparacion.empty or "Es mejor" not in df_comparacion.columns:
        return pd.DataFrame()

    resumen = (
        df_comparacion[df_comparacion["Es mejor"]]
        .copy()
        .sort_values("Producto")
    )

    if resumen.empty:
        return pd.DataFrame()

    return resumen[["Producto", "Método", "wMAPE", "Bias", "MAE"]].rename(
        columns={"Método": "Mejor método"}
    )


def calcular_metricas_ejecutivas(
    df_real: pd.DataFrame,
    df_tvu: pd.DataFrame,
    kpis_tvu: dict,
    resumen_mejores: pd.DataFrame,
):
    total_skus_demanda = df_real["product_id"].nunique() if not df_real.empty else 0
    total_skus_modelados = len(resumen_mejores) if not resumen_mejores.empty else 0

    valor_tvu_riesgo = float(kpis_tvu.get("valor_riesgo", 0))
    stock_tvu_riesgo = float(kpis_tvu.get("stock_riesgo", 0))
    skus_riesgo = int(kpis_tvu.get("sku_alto", 0)) + int(kpis_tvu.get("sku_medio", 0))

    wmape_promedio = resumen_mejores["wMAPE"].mean() if not resumen_mejores.empty else 0
    bias_promedio = resumen_mejores["Bias"].mean() if not resumen_mejores.empty else 0
    modelo_dominante = (
        resumen_mejores["Mejor método"].mode().iloc[0]
        if not resumen_mejores.empty
        else "Sin datos"
    )

    error_mensual_valorizado = 0.0
    if not resumen_mejores.empty and not df_tvu.empty:
        costos = df_tvu[["product_id", "unit_value"]].drop_duplicates("product_id")
        costos = costos.rename(columns={"product_id": "Producto"})
        tmp = resumen_mejores.merge(costos, on="Producto", how="left")
        tmp["unit_value"] = pd.to_numeric(tmp["unit_value"], errors="coerce").fillna(0)
        tmp["error_valorizado_estimado"] = tmp["MAE"] * tmp["unit_value"]
        error_mensual_valorizado = float(tmp["error_valorizado_estimado"].sum())

    return {
        "total_skus_demanda": total_skus_demanda,
        "total_skus_modelados": total_skus_modelados,
        "valor_tvu_riesgo": valor_tvu_riesgo,
        "stock_tvu_riesgo": stock_tvu_riesgo,
        "skus_riesgo": skus_riesgo,
        "wmape_promedio": wmape_promedio,
        "bias_promedio": bias_promedio,
        "modelo_dominante": modelo_dominante,
        "error_mensual_valorizado": error_mensual_valorizado,
    }


def mostrar_dashboard_ejecutivo(
    df_real: pd.DataFrame,
    df_tvu: pd.DataFrame,
    resumen_vencimientos: pd.DataFrame,
    kpis_tvu: dict,
    df_comparacion: pd.DataFrame,
):
    resumen_mejores = construir_resumen_mejores(df_comparacion)
    metricas = calcular_metricas_ejecutivas(df_real, df_tvu, kpis_tvu, resumen_mejores)

    st.title("📊 Dashboard Ejecutivo")
    st.caption("Vista general del portafolio: demanda, pronóstico, riesgo de vencimiento y criticidad económica.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SKUs con demanda", f"{metricas['total_skus_demanda']:,}")
    c2.metric("SKUs modelados", f"{metricas['total_skus_modelados']:,}")
    c3.metric("Modelo dominante", metricas["modelo_dominante"])
    c4.metric("wMAPE promedio", f"{metricas['wmape_promedio']:.2%}")
    c5.metric("Valor en riesgo", f"S/ {metricas['valor_tvu_riesgo']:,.0f}")

    st.divider()

    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        st.subheader("📈 Distribución de modelos ganadores")
        if resumen_mejores.empty:
            st.info("No hay información de métodos ganadores disponible.")
        else:
            conteo_modelos = resumen_mejores["Mejor método"].value_counts().reset_index()
            conteo_modelos.columns = ["Método", "Cantidad de SKUs"]
            fig_modelos = px.pie(
                conteo_modelos,
                names="Método",
                values="Cantidad de SKUs",
                hole=0.45,
                title="Participación por método seleccionado",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_modelos.update_traces(textposition="inside", textinfo="percent+label")
            fig_modelos.update_layout(margin=dict(t=45, b=0, l=0, r=0))
            st.plotly_chart(fig_modelos, use_container_width=True)

    with col_b:
        st.subheader("⚠️ Riesgo de vencimiento")
        if df_tvu.empty or resumen_vencimientos.empty:
            st.info("No hay información TVU disponible.")
        else:
            st.plotly_chart(
                grafico_cantidad_riesgo(resumen_vencimientos),
                use_container_width=True,
            )

    st.divider()

    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.subheader("💰 Valor económico comprometido")
        if df_tvu.empty or resumen_vencimientos.empty:
            st.info("No hay información económica TVU disponible.")
        else:
            st.plotly_chart(
                grafico_valor_riesgo(resumen_vencimientos),
                use_container_width=True,
            )

    with col_d:
        st.subheader("🏆 Top 10 productos críticos")
        if df_tvu.empty:
            st.info("No hay productos para mostrar.")
        else:
            top_tvu = (
                df_tvu[df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])]
                .sort_values("valor_en_riesgo", ascending=False)
                .head(10)
            )
            if top_tvu.empty:
                st.success("No hay productos en riesgo alto o medio.")
            else:
                st.dataframe(
                    formatear_tvu(top_tvu),
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()

    st.subheader("📌 Lectura ejecutiva")
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("SKUs en riesgo alto/medio", f"{metricas['skus_riesgo']:,}")
    col_e2.metric("Stock en riesgo", f"{metricas['stock_tvu_riesgo']:,.0f}")
    col_e3.metric("Error mensual valorizado estimado", f"S/ {metricas['error_mensual_valorizado']:,.0f}")

    st.info(
        "Este resumen muestra indicadores agregados. Para revisar el detalle por SKU, utiliza los módulos "
        "'Pronósticos e Inventarios' y 'TVU - Productos próximos a vencer'."
    )


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
modo_datos = st.sidebar.radio(
    "Modo de datos",
    ["Generar datos sintéticos", "Subir Excel (Pestañas: Demanda y Datos)"],
)

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)
    df_real = generar_demanda_sintetica(n_productos=n_productos, meses=meses, seed=seed)
    df_parametros = pd.DataFrame()
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo Excel unificado", type=["xlsx", "xls"])
    if archivo is None:
        st.info(
            "Sube un archivo Excel que contenga dos pestañas:\n"
            "1. 'Demanda': Con el historial (date, product_id, demand_real)\n"
            "2. 'Datos': Con el maestro de artículos (GRUPO DE DEMANDA, lead_time, etc.)"
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

    except Exception as e:
        st.error(f"Error procesando el archivo: {str(e)}")
        st.stop()


# =========================================================
# TVU - RIESGO DE VENCIMIENTO
# =========================================================
df_tvu = preparar_tvu(df_parametros)
resumen_vencimientos, kpis_tvu = resumen_tvu(df_tvu)

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
        top_10 = df_tvu[df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])].head(10)

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
# PRONÓSTICO AUTOMÁTICO PARA VISTA GENERAL
# =========================================================
fecha_fin_default = pd.Timestamp("2026-12-01")
df_forecast_auto, df_comparacion = ejecutar_forecast_cache(df_real, fecha_fin_default)

if modulo == "📊 Vista General Ejecutiva":
    mostrar_dashboard_ejecutivo(
        df_real=df_real,
        df_tvu=df_tvu,
        resumen_vencimientos=resumen_vencimientos,
        kpis_tvu=kpis_tvu,
        df_comparacion=df_comparacion,
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

if fecha_fin_pronostico != fecha_fin_default:
    df_forecast_auto, df_comparacion = ejecutar_forecast_cache(df_real, fecha_fin_pronostico)

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

    resumen_mejores = construir_resumen_mejores(df_comparacion)

    col_graf, col_tabla = st.columns([1.2, 1])

    if resumen_mejores.empty:
        st.warning("No se pudo construir el resumen de mejores métodos.")
    else:
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

    st.dataframe(
        df_comp.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

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
