import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def grafico_forecast(df_producto: pd.DataFrame) -> go.Figure:
    metodo = df_producto["method_used"].iloc[0] if "method_used" in df_producto.columns else ""

    df_hist = df_producto[df_producto.get("tipo_periodo", "Histórico") == "Histórico"].copy()
    df_future = df_producto[df_producto.get("tipo_periodo", "Histórico") == "Pronóstico futuro"].copy()

    fig = go.Figure()
    
    # Línea histórica
    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["demand_real"],
            mode="lines+markers",
            name="Demanda real",
            line=dict(color="#1f77b4", width=2)
        )
    )
    
    # Línea de ajuste (entrenamiento)
    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["demand_forecast"],
            mode="lines+markers",
            name=f"Ajuste ({metodo})",
            line=dict(color="#ff7f0e", width=2, dash="dot")
        )
    )

    # Línea de pronóstico futuro
    if not df_future.empty:
        fig.add_trace(
            go.Scatter(
                x=df_future["date"],
                y=df_future["demand_forecast"],
                mode="lines+markers",
                name="Pronóstico futuro",
                line=dict(color="#2ca02c", width=3, dash="dash")
            )
        )

    # Ajustes de diseño para evitar choques visuales
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Unidades",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20), # Márgenes reducidos
        legend=dict(
            orientation="h",   # Leyenda horizontal
            yanchor="top",     # Anclada desde arriba
            y=-0.15,           # Posicionada DEBAJO del eje X (evita chocar con la gráfica)
            xanchor="center",  # Centrada horizontalmente
            x=0.5
        )
    )
    return fig

def grafico_inventario(df_sim: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Nivel de Inventario (Línea principal)
    fig.add_trace(
        go.Scatter(
            x=df_sim["date"], 
            y=df_sim["inventory_level"], 
            name="Inventario Físico", 
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )
    
    # 2. Punto de Reorden (Línea de alerta)
    fig.add_trace(
        go.Scatter(
            x=df_sim["date"],
            y=df_sim["reorder_point_s"],
            name="Punto de Reorden (s)",
            mode="lines",
            line=dict(color="#d62728", dash="dot", width=2), # Rojo punteado para denotar peligro/alerta
        ),
        secondary_y=False,
    )
    
    # 3. Demanda Mensual (Barras de fondo)
    fig.add_trace(
        go.Bar(
            x=df_sim["date"], 
            y=df_sim["demand_real"], 
            name="Demanda Real", 
            marker_color="#b3cde3", 
            opacity=0.6
        ),
        secondary_y=True,
    )

    # 4. Pedidos Generados (Marcadores destacados)
    pedidos = df_sim[df_sim["order_placed"] > 0]
    fig.add_trace(
        go.Scatter(
            x=pedidos["date"],
            y=pedidos["order_placed"],
            name="Pedido Generado",
            mode="markers",
            marker=dict(
                size=12, 
                symbol="triangle-up", 
                color="#ff7f0e", 
                line=dict(width=1, color="DarkSlateGrey")
            ),
        ),
        secondary_y=True,
    )

    # Ajustes de diseño y eliminación de título redundante
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_yaxes(title_text="Unidades en Inventario", secondary_y=False)
    fig.update_yaxes(title_text="Demanda / Tamaño de Pedido", secondary_y=True, showgrid=False)
    
    return fig
    

def grafico_tradeoff(df_opt: pd.DataFrame) -> go.Figure:
    mejor = df_opt.loc[df_opt["total_cost"].idxmin()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Fill Rate (Nivel de Servicio) en el eje secundario (Área sombreada al fondo)
    fig.add_trace(
        go.Scatter(
            x=df_opt["ss_months"], 
            y=df_opt["fill_rate"], 
            mode="lines", 
            name="Nivel de Servicio (Fill Rate)",
            fill='tozeroy',
            line=dict(color="rgba(44, 160, 44, 0.2)", width=0), # Verde muy transparente
            fillcolor="rgba(44, 160, 44, 0.1)"
        ),
        secondary_y=True,
    )

    # 2. Costo de Mantener (Sube con el SS)
    fig.add_trace(
        go.Scatter(
            x=df_opt["ss_months"], 
            y=df_opt["holding_cost"], 
            mode="lines", 
            name="Costo de Mantener",
            line=dict(color="#1f77b4", dash="dash") # Azul punteado
        ),
        secondary_y=False,
    )

    # 3. Costo de Quiebre (Baja con el SS)
    fig.add_trace(
        go.Scatter(
            x=df_opt["ss_months"], 
            y=df_opt["stockout_cost"], 
            mode="lines", 
            name="Costo por Quiebre",
            line=dict(color="#d62728", dash="dash") # Rojo punteado
        ),
        secondary_y=False,
    )

    # 4. Costo Total (La curva en forma de U)
    fig.add_trace(
        go.Scatter(
            x=df_opt["ss_months"], 
            y=df_opt["total_cost"], 
            mode="lines+markers", 
            name="Costo Total",
            line=dict(color="#2ca02c", width=3), # Verde fuerte
            marker=dict(size=8)
        ),
        secondary_y=False,
    )

    # 5. Línea vertical marcando el Óptimo
    fig.add_vline(
        x=int(mejor["ss_months"]),
        line_dash="dot",
        line_color="black",
        annotation_text=f"Óptimo: {int(mejor['ss_months'])} meses",
        annotation_position="top left"
    )

    # Ajustes de diseño
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Formato de los ejes
    fig.update_yaxes(title_text="Costo (S/)", secondary_y=False, tickformat=",.0f")
    fig.update_yaxes(title_text="Fill Rate", secondary_y=True, tickformat=".0%", range=[0, 1.05], showgrid=False)
    fig.update_xaxes(title_text="Meses de Stock de Seguridad (SS)", tickmode='linear')

    return fig

def formatear_comparacion(df_comparacion: pd.DataFrame) -> pd.DataFrame:
    df = df_comparacion.copy()
    df["wMAPE"] = df["wMAPE"].map(lambda x: f"{x:.2%}")
    df["Bias"] = df["Bias"].map(lambda x: f"{x:.2%}")
    df["MAE"] = df["MAE"].map(lambda x: f"{x:,.2f}")
    df["Resultado"] = np.where(df["Es mejor"], "✅ Mejor", "")
    return df[["Producto", "Método", "wMAPE", "Bias", "MAE", "Resultado"]]
