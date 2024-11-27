import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import statsmodels.api as sm
from plotly.subplots import make_subplots

def calculate_elasticities_and_prices(df):
    models_by_productline = {}
    elasticities_by_productline = {}

    for product in df['productLine'].unique():
        df_product = df[df['productLine'] == product]
        df_product['log_price'] = np.log(df_product['priceEach'])
        df_product['log_quantity'] = np.log(df_product['quantityOrdered'])
        
        X = sm.add_constant(df_product['log_price'])
        y = df_product['log_quantity']
        model = sm.OLS(y, X).fit()
        
        models_by_productline[product] = model
        elasticities_by_productline[product] = model.params['log_price']

    current_prices = df.groupby('productLine')['priceEach'].mean()
    optimal_prices = {}

    for product, elasticity in elasticities_by_productline.items():
        if elasticity > 0:
            optimal_price = current_prices[product] * 1.1
        elif elasticity < -1:
            optimal_price = current_prices[product] / (1 + (abs(1 / elasticity))/100)
        elif elasticity >= -1 and elasticity < 0:
            optimal_price = current_prices[product] * (1 + (abs(1 / elasticity))/100)
        else:
            optimal_price = current_prices[product]

        if optimal_price > 0 and not np.isinf(optimal_price):
            optimal_prices[product] = optimal_price
        else:
            optimal_prices[product] = current_prices[product]

    comparison = pd.DataFrame({
        "Precio Actual": current_prices,
        "Precio Óptimo": optimal_prices,
        "Elasticidad": elasticities_by_productline,
    })

    def classify_elasticity(elasticity):
        if elasticity > 0:
            return "Elástica positiva (lujo)"
        elif elasticity < -1:
            return "Elástica"
        elif -1 <= elasticity < 0:
            return "Inelástica"
        else:
            return "Sin cambio"

    comparison['Tipo de Demanda'] = comparison['Elasticidad'].apply(classify_elasticity)
    comparison['Variación Porcentual'] = ((comparison['Precio Óptimo'] - comparison['Precio Actual']) / comparison['Precio Actual']) * 100

    return comparison, models_by_productline

def create_dashboard():
    st.set_page_config(
        page_title="Análisis de Elasticidades y Precios Óptimos",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Elasticity Analysis and Optimal Prices Dashboard")
    
    try:
        df = pd.read_csv('data.csv')
        st.success("✅ Data loaded successfully")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    comparison, models = calculate_elasticities_and_prices(df)

    # Análisis de ingresos
    revenue_analysis = pd.DataFrame(index=comparison.index)
    revenue_analysis['Cantidad Actual'] = df.groupby('productLine')['quantityOrdered'].sum()
    revenue_analysis['Precio Actual'] = comparison['Precio Actual']
    revenue_analysis['Precio Óptimo'] = comparison['Precio Óptimo']
    revenue_analysis['Ingreso Actual'] = revenue_analysis['Cantidad Actual'] * revenue_analysis['Precio Actual']
    revenue_analysis['Elasticidad'] = comparison['Elasticidad']
    revenue_analysis['Variación % Precio'] = comparison['Variación Porcentual']
    revenue_analysis['Variación % Cantidad'] = revenue_analysis['Variación % Precio'] * revenue_analysis['Elasticidad']
    revenue_analysis['Cantidad Proyectada'] = revenue_analysis['Cantidad Actual'] * (1 + revenue_analysis['Variación % Cantidad']/100)
    revenue_analysis['Ingreso Proyectado'] = revenue_analysis['Cantidad Proyectada'] * revenue_analysis['Precio Óptimo']
    revenue_analysis['Variación Ingreso'] = revenue_analysis['Ingreso Proyectado'] - revenue_analysis['Ingreso Actual']
    revenue_analysis['Variación % Ingreso'] = (revenue_analysis['Variación Ingreso'] / revenue_analysis['Ingreso Actual']) * 100

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_current_revenue = revenue_analysis['Ingreso Actual'].sum()
    total_projected_revenue = revenue_analysis['Ingreso Proyectado'].sum()
    total_revenue_change = total_projected_revenue - total_current_revenue
    
    with col1:
        st.metric(
            "Total Products",
            len(df['productLine'].unique())
        )
    with col2:
        st.metric(
            "Total Current Income",
            f"${total_current_revenue:,.2f}"
        )
    with col3:
        st.metric(
            "Total Projected Income",
            f"${total_projected_revenue:,.2f}"
        )
    with col4:
        st.metric(
            "Total Variation",
            f"${total_revenue_change:,.2f}",
            delta=f"{(total_revenue_change/total_current_revenue)*100:.1f}%"
        )

    # Tabs para visualizaciones
    tab1, tab2, tab3 = st.tabs(["📈 Prices & Elasticities", "💰 Revenue Analysis", "📊 Detailed Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_prices = go.Figure()
            fig_prices.add_trace(go.Bar(
                name='Current Price',
                x=comparison.index,
                y=comparison['Precio Actual'],
                marker_color='lightblue'
            ))
            fig_prices.add_trace(go.Bar(
                name='Optimal Price',
                x=comparison.index,
                y=comparison['Precio Óptimo'],
                marker_color='darkblue'
            ))
            fig_prices.update_layout(
                title="Price Comparison",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            fig_elasticity = go.Figure()
            fig_elasticity.add_trace(go.Bar(
                x=comparison.index,
                y=comparison['Elasticidad'],
                marker_color=['red' if x < -1 else 'green' if x > 0 else 'yellow' 
                             for x in comparison['Elasticidad']],
                text=comparison['Elasticidad'].round(2),
                textposition='auto',
            ))
            fig_elasticity.update_layout(
                title="Products Elasticities",
                height=400
            )
            st.plotly_chart(fig_elasticity, use_container_width=True)

    with tab2:
        # Gráfico de ingresos
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            name='Current Income',
            x=revenue_analysis.index,
            y=revenue_analysis['Ingreso Actual'],
            marker_color='lightblue'
        ))
        fig_revenue.add_trace(go.Bar(
            name='Projected Income',
            x=revenue_analysis.index,
            y=revenue_analysis['Ingreso Proyectado'],
            marker_color='darkblue'
        ))
        fig_revenue.update_layout(
            title="Income Comparison",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

        # Gráfico de waterfall
        fig_waterfall = go.Figure(go.Waterfall(
            name="Income Variation",
            orientation="v",
            measure=["relative"] * len(revenue_analysis.index) + ["total"],
            x=[*revenue_analysis.index, "Total"],
            textposition="outside",
            text=[f"${x:,.0f}" for x in revenue_analysis['Variación Ingreso']] + [f"${total_revenue_change:,.0f}"],
            y=[*revenue_analysis['Variación Ingreso'], total_revenue_change],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"green"}},
            decreasing={"marker":{"color":"red"}},
            totals={"marker":{"color":"blue"}}
        ))
        fig_waterfall.update_layout(
            title="Impact on Revenue by Product",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab3:
        # Tabla de datos detallados
        st.subheader("Detailed Analysis by Product")
        detailed_df = pd.DataFrame({
            'Current Price': revenue_analysis['Precio Actual'],
            'Optimal Price': revenue_analysis['Precio Óptimo'],
            'Current Quantity': revenue_analysis['Cantidad Actual'],
            'Projected Quantity': revenue_analysis['Cantidad Proyectada'],
            'Current Income': revenue_analysis['Ingreso Actual'],
            'Projected Income': revenue_analysis['Ingreso Proyectado'],
            'Income Variation': revenue_analysis['Variación Ingreso'],
            'Elasticity': revenue_analysis['Elasticidad'],
            'Variation % Price': revenue_analysis['Variación % Precio'],
            'Variation % Quantity': revenue_analysis['Variación % Cantidad'],
            'Variation % Income': revenue_analysis['Variación % Ingreso']
        }).round(2)

        # Formatear columnas monetarias
        for col in ['Current Price', 'Optimal Price', 'Current Income', 'Projected Income', 'Income Variation']:
            detailed_df[col] = detailed_df[col].apply(lambda x: f"${x:,.2f}")

        # Formatear columnas porcentuales
        for col in ['Variation % Price', 'Variation % Quantity', 'Variation % Income']:
            detailed_df[col] = detailed_df[col].apply(lambda x: f"{x:,.2f}%")

        # Formatear cantidades
        for col in ['Current Quantity', 'Projected Quantity']:
            detailed_df[col] = detailed_df[col].apply(lambda x: f"{x:,.0f}")

        st.dataframe(detailed_df, use_container_width=True)

    # Recomendaciones
    st.markdown("### 💡 Strategic Recommendations")
    
    for product in comparison.index:
        elasticity = comparison.loc[product, 'Elasticidad']
        var_pct = comparison.loc[product, 'Variación Porcentual']
        var_ingreso = revenue_analysis.loc[product, 'Variación Ingreso']
        var_ingreso_pct = revenue_analysis.loc[product, 'Variación % Ingreso']
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if elasticity < -1:
                st.error(f"📉 {product}")
            elif elasticity > 0:
                st.success(f"📈 {product}")
            else:
                st.info(f"📊 {product}")
        
        with col2:
            if elasticity < -1:
                st.markdown(f"""
                    - Elasticity: {elasticity:.2f} (High price sensitivity)
                    - Recommended price change: {var_pct:.1f}%
                    - Impact on revenue: ${var_ingreso:,.2f} ({var_ingreso_pct:.1f}%)
                    - Strategy: Reduce prices to increase sales
                """)
            elif elasticity > 0:
                st.markdown(f"""
                    - Elasticity: {elasticity:.2f} (Luxury product)
                    - Recommended price change: {var_pct:.1f}%
                    - Impact on revenue: ${var_ingreso:,.2f} ({var_ingreso_pct:.1f}%)
                    - Strategy: Increase prices and emphasize exclusivity
                """)
            else:
                st.markdown(f"""
                    - Elasticity: {elasticity:.2f} (Inelastic demand)
                    - Recommended price change: {var_pct:.1f}%
                    - Impact on revenue: ${var_ingreso:,.2f} ({var_ingreso_pct:.1f}%)
                    - Strategy: Moderate price adjustment
                """)

if __name__ == "__main__":
    create_dashboard()