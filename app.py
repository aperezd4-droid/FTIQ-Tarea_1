import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador de Viscosidad", layout="wide")
st.title("🧪 Simulador de Viscosidad de Gases")

# ==========================================
# 1. BASE DE DATOS (PROPIEDADES Y EXPERIMENTAL)
# ==========================================
datos_componentes = {
    'Componente': ['H2', 'CH4', 'CO', 'CO2', 'N2', 'C2H4', 'C2H6', 'C3H6', 'H2S'],
    'M (g/mol)': [2.016, 16.043, 28.01, 44.009, 28.014, 28.054, 30.07, 42.08, 34.081],
    'σ (Å)': [2.827, 3.758, 3.69, 3.941, 3.798, 4.163, 4.443, 4.678, 3.623],
    'ε/k (K)': [59.7, 148.6, 91.7, 195.2, 71.4, 224.7, 215.7, 298.9, 301.1],
    'T_c (K)': [33.2, 190.6, 132.9, 304.2, 126.2, 282.3, 305.3, 365.0, 373.2], # NUEVA COLUMNA
    'V_c (cm3/mol)': [65.0, 99.2, 93.2, 94.0, 89.9, 131.0, 148.0, 181.0, 98.5],
    'T_exp_default (K)': [323.13, 323.86, 300.0, 373.15, 323.15, 300.0, 323.15, 323.15, 323.96],
    'P_exp_default (MPa)': [1.69, 0.1, 0.05, 1.013, 1.50, 1.0, 1.013, 1.43, 0.05]
}
df_comp = pd.DataFrame(datos_componentes)

datos_exp = {
    'Componente': ['H2', 'CH4', 'CO', 'CO2', 'N2', 'C2H4', 'C2H6', 'C3H6', 'H2S'],
    'Viscosidad Exp (Pa*s)': [9.42e-6, 1.19e-5, 1.78e-5, 1.86e-5, 1.91e-5, 1.06e-5, 9.99e-6, 9.79e-6, 1.32e-5]
}
df_exp = pd.DataFrame(datos_exp)

# ==========================================
# 2. FUNCIONES MATEMÁTICAS
# ==========================================
def modelo_1_chapman_enskog(T, M, sigma, epsilon_k):
    T_star = T / epsilon_k
    term1 = 1.16145 / (T_star ** 0.14874)
    term2 = 0.52487 / np.exp(0.77320 * T_star)
    term3 = 2.16178 / np.exp(2.43787 * T_star)
    omega_v = term1 + term2 + term3
    
    viscosidad_uP = 26.69 * np.sqrt(M * T) / ((sigma ** 2) * omega_v)
    return viscosidad_uP * 1e-7, T_star, omega_v

def modelo_2_estados_correspondientes(T, M, V_c, T_c):
    # Calculamos T_r (Temperatura reducida)
    T_r = T / T_c
    
    # 1. Cálculo de n* (Viscosidad reducida)
    term1 = 0.807 * (T_r ** 0.618)
    term2 = 0.357 * np.exp(-0.449 * T_r)
    term3 = 0.34 * np.exp(-4.058 * T_r)
    eta_star = term1 - term2 + term3 + 0.018
    
    # 2. Cálculo de la viscosidad final (Pa*s) usando T_c y no T
    viscosidad_Pa_s = (4.0785e-6) * eta_star * (math.sqrt(M * T_c) / (V_c ** (2/3)))
    
    return viscosidad_Pa_s, eta_star, T_r

# ==========================================
# 3. INTERFAZ: PANEL LATERAL DINÁMICO
# ==========================================
st.sidebar.header("⚙️ Condiciones de Operación")
st.sidebar.markdown("Ajusta T y P para cada gas individualmente:")

condiciones_usuario = {}
for index, row in df_comp.iterrows():
    comp = row['Componente']
    st.sidebar.markdown(f"**{comp}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        t_val = st.number_input("T (K)", value=float(row['T_exp_default (K)']), step=1.0, key=f"t_{comp}")
    with col2:
        p_val = st.number_input("P (MPa)", value=float(row['P_exp_default (MPa)']), step=0.1, key=f"p_{comp}")
    condiciones_usuario[comp] = {'T': t_val, 'P': p_val}
    st.sidebar.markdown("---")

# ==========================================
# 4. PESTAÑAS Y CÁLCULOS
# ==========================================
tab1, tab2 = st.tabs(["📘 Modelo 1: Chapman-Enskog", "📗 Modelo 2: Estados Correspondientes"])

# --- PESTAÑA MODELO 1 ---
with tab1:
    st.header("Resultados Modelo 1 (Chapman-Enskog)")
    resultados_m1 = []
    for index, row in df_comp.iterrows():
        comp = row['Componente']
        T_user = condiciones_usuario[comp]['T']
        P_user = condiciones_usuario[comp]['P']
        
        visc_Pa_s, T_star, omega_v = modelo_1_chapman_enskog(T_user, row['M (g/mol)'], row['σ (Å)'], row['ε/k (K)'])
        resultados_m1.append({
            'Componente': comp, 'T (K)': T_user, 'P (MPa)': P_user,
            'T*': round(T_star, 4), 'Ω_v': round(omega_v, 4), 'Viscosidad Calc (Pa*s)': visc_Pa_s
        })
        
    df_res_m1 = pd.DataFrame(resultados_m1)
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m1.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion = pd.merge(df_exp, df_res_m1, on='Componente')
    df_comparacion = df_comparacion[['Componente', 'T (K)', 'P (MPa)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)']]
    df_comparacion['% Error'] = abs(df_comparacion['Viscosidad Calc (Pa*s)'] - df_comparacion['Viscosidad Exp (Pa*s)']) / df_comparacion['Viscosidad Exp (Pa*s)'] * 100
    st.dataframe(df_comparacion.style.format({'Viscosidad Exp (Pa*s)': "{:.4e}", 'Viscosidad Calc (Pa*s)': "{:.4e}", '% Error': "{:.2f} %"}), use_container_width=True)

    st.divider()
    y_true = df_comparacion['Viscosidad Exp (Pa*s)']
    y_pred = df_comparacion['Viscosidad Calc (Pa*s)']
    r2_score = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    mape = np.mean(df_comparacion['% Error'])

    col_grafica, col_metricas = st.columns([3, 1])
    with col_grafica:
        st.markdown("**Gráfica de Paridad: Modelo 1 vs Experimental**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true, y_pred, color='dodgerblue', edgecolor='black', s=80, label='Predicción M1', zorder=3)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (Error 0%)', zorder=2)
        ax.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax.set_ylabel('Viscosidad Modelo 1 (Pa*s)', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.7, zorder=1)
        ax.legend()
        st.pyplot(fig)

    with col_metricas:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("📈 **Métricas de Ajuste**")
        st.metric(label="Coeficiente R²", value=f"{r2_score:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{mape:.2f} %")

# --- PESTAÑA MODELO 2 ---
with tab2:
    st.header("Resultados Modelo 2 (Estados Correspondientes)")
    
    resultados_m2 = []
    for index, row in df_comp.iterrows():
        comp = row['Componente']
        T_user = condiciones_usuario[comp]['T']
        P_user = condiciones_usuario[comp]['P']
        
        visc_Pa_s, eta_star, T_r = modelo_2_estados_correspondientes(T_user, row['M (g/mol)'], row['V_c (cm3/mol)'], row['T_c (K)'])
        resultados_m2.append({
            'Componente': comp, 'T (K)': T_user, 'T_c (K)': row['T_c (K)'], 'T_r': round(T_r, 4),
            'η* (reducida)': round(eta_star, 4), 'Viscosidad Calc (Pa*s)': visc_Pa_s
        })
        
    df_res_m2 = pd.DataFrame(resultados_m2)
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m2.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion_m2 = pd.merge(df_exp, df_res_m2, on='Componente')
    df_comparacion_m2 = df_comparacion_m2[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)']]
    df_comparacion_m2['% Error'] = abs(df_comparacion_m2['Viscosidad Calc (Pa*s)'] - df_comparacion_m2['Viscosidad Exp (Pa*s)']) / df_comparacion_m2['Viscosidad Exp (Pa*s)'] * 100
    st.dataframe(df_comparacion_m2.style.format({'Viscosidad Exp (Pa*s)': "{:.4e}", 'Viscosidad Calc (Pa*s)': "{:.4e}", '% Error': "{:.2f} %"}), use_container_width=True)

    st.divider()
    
    y_true_m2 = df_comparacion_m2['Viscosidad Exp (Pa*s)']
    y_pred_m2 = df_comparacion_m2['Viscosidad Calc (Pa*s)']
    r2_score_m2 = 1 - (np.sum((y_true_m2 - y_pred_m2) ** 2) / np.sum((y_true_m2 - np.mean(y_true_m2)) ** 2))
    mape_m2 = np.mean(df_comparacion_m2['% Error'])

    col_grafica_m2, col_metricas_m2 = st.columns([3, 1])
    with col_grafica_m2:
        st.markdown("**Gráfica de Paridad: Modelo 2 vs Experimental**")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_true_m2, y_pred_m2, color='mediumseagreen', edgecolor='black', s=80, label='Predicción M2', zorder=3)
        min_val_m2, max_val_m2 = min(y_true_m2.min(), y_pred_m2.min()), max(y_true_m2.max(), y_pred_m2.max())
        ax2.plot([min_val_m2, max_val_m2], [min_val_m2, max_val_m2], color='red', linestyle='--', label='Ideal (Error 0%)', zorder=2)
        ax2.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax2.set_ylabel('Viscosidad Modelo 2 (Pa*s)', fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.7, zorder=1)
        ax2.legend()
        st.pyplot(fig2)

    with col_metricas_m2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("📈 **Métricas de Ajuste**")
        st.metric(label="Coeficiente R²", value=f"{r2_score_m2:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{mape_m2:.2f} %")