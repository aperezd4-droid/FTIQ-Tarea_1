import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador de Viscosidad", layout="wide")
st.title("🧪 Simulador de Viscosidad de Gases y Mezclas")

# ==========================================
# 1. BASE DE DATOS (PROPIEDADES Y EXPERIMENTAL)
# ==========================================
datos_componentes = {
    'Componente': ['H2', 'CH4', 'CO', 'CO2', 'N2', 'C2H4', 'C2H6', 'C3H6', 'H2S'],
    'M (g/mol)': [2.016, 16.043, 28.01, 44.009, 28.014, 28.054, 30.07, 42.08, 34.081],
    'σ (Å)': [2.915, 3.78, 3.59, 3.996, 3.667, 4.228, 4.766, 4.678, 3.623],
    'ε/k (K)': [38, 154, 110, 190, 99.8, 216, 275, 298.9, 301.1],
    'T_c (K)': [33.2, 190.6, 132.9, 304.2, 126.2, 282.3, 305.3, 365.0, 373.2],
    'P_c (atm)': [12.8, 45.4, 34.5, 72.8, 33.5, 49.7, 48.2, 45.6, 88.2], 
    'V_c (cm3/mol)': [65.0, 99.2, 93.2, 94.0, 89.9, 131.0, 148.0, 181.0, 98.5],
    'η_0 (Pa*s)': [0.00000876, 0.000011, 0.0000172, 0.0000148, 0.00001781, 0.0000092, 0.0000093, 0.0000083, 0.000012], 
    'T_0 (K)': [293.85, 300.0, 288.15, 293.15, 300.55, 300.0, 300.0, 300.0, 293.0], 
    'S (K)': [72.0, 198.0, 102.0, 240.0, 107.0, 226.0, 252.0, 298.0, 331.0], 
    'T_exp_default (K)': [323.13, 323.86, 300.0, 373.15, 323.15, 300.0, 323.15, 323.15, 323.96],
    'P_exp_default (MPa)': [1.69, 0.1, 0.05, 1.013, 1.50, 1.0, 1.013, 1.43, 0.05],
    'X_i_raw': [0.575, 0.25, 0.065, 0.015, 0.045, 0.0125, 0.0065, 0.00035, 0.00000016]
}
df_comp = pd.DataFrame(datos_componentes)

# Normalización de fracciones para la mezcla
suma_X = df_comp['X_i_raw'].sum()
df_comp['y_i (Normalizado)'] = df_comp['X_i_raw'] / suma_X

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
    T_r = T / T_c
    term1 = 0.807 * (T_r ** 0.618)
    term2 = 0.357 * np.exp(-0.449 * T_r)
    term3 = 0.34 * np.exp(-4.058 * T_r)
    eta_star = term1 - term2 + term3 + 0.018
    viscosidad_Pa_s = (8.45e-7) * eta_star * (math.sqrt(M * T_c) / (V_c ** (2/3)))
    return viscosidad_Pa_s, eta_star, T_r

def modelo_3_sutherland(T, eta_0, T_0, S):
    viscosidad_Pa_s = eta_0 * ((T / T_0) ** 1.5) * ((T_0 + S) / (T + S))
    return viscosidad_Pa_s

def modelo_4_criticos(T, M, T_c, P_c, V_c):
    T_r = T / T_c
    xi = (T_c ** (1/6)) / (math.sqrt(M) * (P_c ** (2/3)))
    num_eta = (0.807 * (T_r ** 0.618)) - (0.357 * np.exp(-0.449 * T_r)) + (0.34 * np.exp(-4.058 * T_r)) + 0.018
    eta_r = num_eta / xi
    eta_uP = 10 * eta_r * (math.sqrt(M * T_c) / (V_c ** (2/3)))
    return eta_uP * 1e-7, T_r, xi, eta_r, eta_uP

def regla_mezcla_wilke(y_array, eta_array, M_array):
    n = len(y_array)
    phi = np.zeros((n, n))
    
    # Calcular matriz de interacción Phi
    for i in range(n):
        for j in range(n):
            if i == j:
                phi[i, j] = 1.0
            else:
                num = (1 + math.sqrt(eta_array[i] / eta_array[j]) * ((M_array[j] / M_array[i]) ** 0.25)) ** 2
                den = math.sqrt(8 * (1 + (M_array[i] / M_array[j])))
                phi[i, j] = num / den
                
    # Calcular viscosidad de la mezcla
    eta_mezcla = 0
    for i in range(n):
        suma_denominador = sum(y_array[j] * phi[i, j] for j in range(n))
        eta_mezcla += (y_array[i] * eta_array[i]) / suma_denominador
        
    return eta_mezcla, phi

# ==========================================
# 3. INTERFAZ: PANEL LATERAL DINÁMICO
# ==========================================
st.sidebar.header("⚙️ Condiciones de Operación")
st.sidebar.markdown("Ajusta T y P para cada gas individualmente:")

condiciones_usuario = {}
resultados_m1_global = []
resultados_m2_global = []
resultados_m3_global = []
resultados_m4_global = []

# Listas para guardar viscosidades puras y usarlas en mezclas
visc_m1_list = []
visc_m2_list = []
visc_m3_list = []
visc_m4_list = []

for index, row in df_comp.iterrows():
    comp = row['Componente']
    st.sidebar.markdown(f"**{comp}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        t_val = st.number_input("T (K)", value=float(row['T_exp_default (K)']), step=1.0, key=f"t_{comp}")
    with col2:
        p_val = st.number_input("P (MPa)", value=float(row['P_exp_default (MPa)']), step=0.1, key=f"p_{comp}")
    condiciones_usuario[comp] = {'T': t_val, 'P': p_val}
    
    # Cálculos globales para todos los modelos
    # M1
    visc_m1, T_star, omega_v = modelo_1_chapman_enskog(t_val, row['M (g/mol)'], row['σ (Å)'], row['ε/k (K)'])
    resultados_m1_global.append({'Componente': comp, 'T (K)': t_val, 'T*': round(T_star, 4), 'Ω_v': round(omega_v, 4), 'Viscosidad Calc (Pa*s)': visc_m1})
    visc_m1_list.append(visc_m1)
    
    # M2
    visc_m2, eta_star, T_r_m2 = modelo_2_estados_correspondientes(t_val, row['M (g/mol)'], row['V_c (cm3/mol)'], row['T_c (K)'])
    resultados_m2_global.append({'Componente': comp, 'T (K)': t_val, 'T_r': round(T_r_m2, 4), 'η*': round(eta_star, 4), 'Viscosidad Calc (Pa*s)': visc_m2})
    visc_m2_list.append(visc_m2)
    
    # M3
    visc_m3 = modelo_3_sutherland(t_val, row['η_0 (Pa*s)'], row['T_0 (K)'], row['S (K)'])
    resultados_m3_global.append({'Componente': comp, 'T (K)': t_val, 'η_0 (Pa*s)': row['η_0 (Pa*s)'], 'Viscosidad Calc (Pa*s)': visc_m3})
    visc_m3_list.append(visc_m3)
    
    # M4
    visc_m4, T_r_m4, xi, eta_r, eta_uP = modelo_4_criticos(t_val, row['M (g/mol)'], row['T_c (K)'], row['P_c (atm)'], row['V_c (cm3/mol)'])
    resultados_m4_global.append({'Componente': comp, 'T (K)': t_val, 'T_r': round(T_r_m4, 4), 'ξ (Xi)': round(xi, 5), 'η_r': round(eta_r, 4), 'η (μP)': round(eta_uP, 2), 'Viscosidad Calc (Pa*s)': visc_m4})
    visc_m4_list.append(visc_m4)
    
    st.sidebar.markdown("---")

df_res_m1 = pd.DataFrame(resultados_m1_global)
df_res_m2 = pd.DataFrame(resultados_m2_global)
df_res_m3 = pd.DataFrame(resultados_m3_global)
df_res_m4 = pd.DataFrame(resultados_m4_global)

# Diccionario para facilitar la selección en las mezclas
opciones_modelos = {
    "Modelo 1 (Chapman-Enskog)": visc_m1_list,
    "Modelo 2 (Estados Correspondientes)": visc_m2_list,
    "Modelo 3 (Sutherland)": visc_m3_list,
    "Modelo 4 (Yoon-Thodos)": visc_m4_list
}

# ==========================================
# 4. PESTAÑAS Y CÁLCULOS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 M1: Chapman-Enskog", 
    "📗 M2: Est. Correspondientes", 
    "📙 M3: Sutherland", 
    "🟪 M4: Yoon-Thodos",
    "⚗️ M5: Mezcla Wilke",
    "🧪 M6: Mezcla Herning-Zipperer"
])

# --- PESTAÑA MODELO 1 ---
with tab1:
    st.header("Resultados Modelo 1 (Chapman-Enskog)")
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m1.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion = pd.merge(df_exp, df_res_m1, on='Componente')
    df_comparacion['% Error'] = abs(df_comparacion['Viscosidad Calc (Pa*s)'] - df_comparacion['Viscosidad Exp (Pa*s)']) / df_comparacion['Viscosidad Exp (Pa*s)'] * 100
    
    # MOSTRAR LA TABLA DE COMPARACIÓN
    st.dataframe(df_comparacion[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)', '% Error']].style.format({
        'Viscosidad Exp (Pa*s)': "{:.4e}", 
        'Viscosidad Calc (Pa*s)': "{:.4e}", 
        '% Error': "{:.2f} %"
    }), use_container_width=True)
    
    y_true = df_comparacion['Viscosidad Exp (Pa*s)']
    y_pred = df_comparacion['Viscosidad Calc (Pa*s)']
    r2_score = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    col_grafica, col_metricas = st.columns([3, 1])
    with col_grafica:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true, y_pred, color='dodgerblue', edgecolor='black', s=80, label='Predicción M1', zorder=3)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal', zorder=2)
        ax.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax.set_ylabel('Viscosidad Modelo 1 (Pa*s)', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
    with col_metricas:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric(label="Coeficiente R²", value=f"{r2_score:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comparacion['% Error']):.2f} %")

# --- PESTAÑA MODELO 2 ---
with tab2:
    st.header("Resultados Modelo 2 (Estados Correspondientes)")
    resultados_m2 = []
    for index, row in df_comp.iterrows():
        comp = row['Componente']
        T_user = condiciones_usuario[comp]['T']
        visc_Pa_s, eta_star, T_r = modelo_2_estados_correspondientes(T_user, row['M (g/mol)'], row['V_c (cm3/mol)'], row['T_c (K)'])
        resultados_m2.append({'Componente': comp, 'T (K)': T_user, 'T_r': round(T_r, 4), 'η*': round(eta_star, 4), 'Viscosidad Calc (Pa*s)': visc_Pa_s})
        
    df_res_m2 = pd.DataFrame(resultados_m2)
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m2.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion_m2 = pd.merge(df_exp, df_res_m2, on='Componente')
    df_comparacion_m2['% Error'] = abs(df_comparacion_m2['Viscosidad Calc (Pa*s)'] - df_comparacion_m2['Viscosidad Exp (Pa*s)']) / df_comparacion_m2['Viscosidad Exp (Pa*s)'] * 100
    
    # MOSTRAR LA TABLA DE COMPARACIÓN
    st.dataframe(df_comparacion_m2[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)', '% Error']].style.format({
        'Viscosidad Exp (Pa*s)': "{:.4e}", 
        'Viscosidad Calc (Pa*s)': "{:.4e}", 
        '% Error': "{:.2f} %"
    }), use_container_width=True)

    y_true_m2 = df_comparacion_m2['Viscosidad Exp (Pa*s)']
    y_pred_m2 = df_comparacion_m2['Viscosidad Calc (Pa*s)']
    r2_score_m2 = 1 - (np.sum((y_true_m2 - y_pred_m2) ** 2) / np.sum((y_true_m2 - np.mean(y_true_m2)) ** 2))

    col_grafica_m2, col_metricas_m2 = st.columns([3, 1])
    with col_grafica_m2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_true_m2, y_pred_m2, color='mediumseagreen', edgecolor='black', s=80, label='Predicción M2', zorder=3)
        min_val_m2, max_val_m2 = min(y_true_m2.min(), y_pred_m2.min()), max(y_true_m2.max(), y_pred_m2.max())
        ax2.plot([min_val_m2, max_val_m2], [min_val_m2, max_val_m2], color='red', linestyle='--', label='Ideal', zorder=2)
        ax2.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax2.set_ylabel('Viscosidad Modelo 2 (Pa*s)', fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.legend()
        st.pyplot(fig2)
    with col_metricas_m2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric(label="Coeficiente R²", value=f"{r2_score_m2:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comparacion_m2['% Error']):.2f} %")

# --- PESTAÑA MODELO 3 ---
with tab3:
    st.header("Resultados Modelo 3 (Sutherland)")
    resultados_m3 = []
    for index, row in df_comp.iterrows():
        comp = row['Componente']
        T_user = condiciones_usuario[comp]['T']
        visc_Pa_s = modelo_3_sutherland(T_user, row['η_0 (Pa*s)'], row['T_0 (K)'], row['S (K)'])
        resultados_m3.append({'Componente': comp, 'T (K)': T_user, 'η_0 (Pa*s)': row['η_0 (Pa*s)'], 'Viscosidad Calc (Pa*s)': visc_Pa_s})
        
    df_res_m3 = pd.DataFrame(resultados_m3)
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m3.style.format({'η_0 (Pa*s)': "{:.4e}", 'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion_m3 = pd.merge(df_exp, df_res_m3, on='Componente')
    df_comparacion_m3['% Error'] = abs(df_comparacion_m3['Viscosidad Calc (Pa*s)'] - df_comparacion_m3['Viscosidad Exp (Pa*s)']) / df_comparacion_m3['Viscosidad Exp (Pa*s)'] * 100
    
    # MOSTRAR LA TABLA DE COMPARACIÓN
    st.dataframe(df_comparacion_m3[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)', '% Error']].style.format({
        'Viscosidad Exp (Pa*s)': "{:.4e}", 
        'Viscosidad Calc (Pa*s)': "{:.4e}", 
        '% Error': "{:.2f} %"
    }), use_container_width=True)

    y_true_m3 = df_comparacion_m3['Viscosidad Exp (Pa*s)']
    y_pred_m3 = df_comparacion_m3['Viscosidad Calc (Pa*s)']
    r2_score_m3 = 1 - (np.sum((y_true_m3 - y_pred_m3) ** 2) / np.sum((y_true_m3 - np.mean(y_true_m3)) ** 2))

    col_grafica_m3, col_metricas_m3 = st.columns([3, 1])
    with col_grafica_m3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(y_true_m3, y_pred_m3, color='darkorange', edgecolor='black', s=80, label='Predicción M3', zorder=3)
        min_val_m3, max_val_m3 = min(y_true_m3.min(), y_pred_m3.min()), max(y_true_m3.max(), y_pred_m3.max())
        ax3.plot([min_val_m3, max_val_m3], [min_val_m3, max_val_m3], color='red', linestyle='--', label='Ideal', zorder=2)
        ax3.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax3.set_ylabel('Viscosidad Modelo 3 (Pa*s)', fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.7)
        ax3.legend()
        st.pyplot(fig3)
    with col_metricas_m3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric(label="Coeficiente R²", value=f"{r2_score_m3:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comparacion_m3['% Error']):.2f} %")

# --- PESTAÑA MODELO 4 ---
with tab4:
    st.header("Resultados Modelo 4 (Yoon-Thodos)")
    resultados_m4 = []
    for index, row in df_comp.iterrows():
        comp = row['Componente']
        T_user = condiciones_usuario[comp]['T']
        visc_Pa_s, T_r, xi, eta_r, eta_uP = modelo_4_criticos(T_user, row['M (g/mol)'], row['T_c (K)'], row['P_c (atm)'], row['V_c (cm3/mol)'])
        resultados_m4.append({'Componente': comp, 'T (K)': T_user, 'T_r': round(T_r, 4), 'ξ (Xi)': round(xi, 5), 'η_r': round(eta_r, 4), 'η (μP)': round(eta_uP, 2), 'Viscosidad Calc (Pa*s)': visc_Pa_s})
        
    df_res_m4 = pd.DataFrame(resultados_m4)
    st.subheader("📋 Tabla de Resultados Calculados")
    st.dataframe(df_res_m4.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}"}), use_container_width=True)

    st.divider()
    st.subheader("📊 Comparación con Datos Experimentales")
    df_comparacion_m4 = pd.merge(df_exp, df_res_m4, on='Componente')
    df_comparacion_m4['% Error'] = abs(df_comparacion_m4['Viscosidad Calc (Pa*s)'] - df_comparacion_m4['Viscosidad Exp (Pa*s)']) / df_comparacion_m4['Viscosidad Exp (Pa*s)'] * 100
    
    # MOSTRAR LA TABLA DE COMPARACIÓN
    st.dataframe(df_comparacion_m4[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)', '% Error']].style.format({
        'Viscosidad Exp (Pa*s)': "{:.4e}", 
        'Viscosidad Calc (Pa*s)': "{:.4e}", 
        '% Error': "{:.2f} %"
    }), use_container_width=True)

    y_true_m4 = df_comparacion_m4['Viscosidad Exp (Pa*s)']
    y_pred_m4 = df_comparacion_m4['Viscosidad Calc (Pa*s)']
    r2_score_m4 = 1 - (np.sum((y_true_m4 - y_pred_m4) ** 2) / np.sum((y_true_m4 - np.mean(y_true_m4)) ** 2))

    col_grafica_m4, col_metricas_m4 = st.columns([3, 1])
    with col_grafica_m4:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.scatter(y_true_m4, y_pred_m4, color='mediumpurple', edgecolor='black', s=80, label='Predicción M4', zorder=3)
        min_val_m4, max_val_m4 = min(y_true_m4.min(), y_pred_m4.min()), max(y_true_m4.max(), y_pred_m4.max())
        ax4.plot([min_val_m4, max_val_m4], [min_val_m4, max_val_m4], color='red', linestyle='--', label='Ideal', zorder=2)
        ax4.set_xlabel('Viscosidad Experimental (Pa*s)', fontweight='bold')
        ax4.set_ylabel('Viscosidad Modelo 4 (Pa*s)', fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.7)
        ax4.legend()
        st.pyplot(fig4)
    with col_metricas_m4:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric(label="Coeficiente R²", value=f"{r2_score_m4:.4f}")
        st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comparacion_m4['% Error']):.2f} %")

# --- PESTAÑA MEZCLA WILKE (NUEVA) ---
with tab5:
    st.header("⚗️ Viscosidad de la Mezcla (Regla de Wilke)")
    st.markdown("Cálculo basado en la matriz de interacción de Wilke (1950) para sistemas multicomponente.")
    
    modelo_seleccionado_wilke = st.selectbox(
        "Elige el modelo para las viscosidades puras (η_i):",
        list(opciones_modelos.keys()),
        key="select_wilke"
    )
    
    viscosidades_base_wilke = opciones_modelos[modelo_seleccionado_wilke]
    
    df_mezcla_w = df_comp[['Componente', 'M (g/mol)', 'X_i_raw', 'y_i (Normalizado)']].copy()
    df_mezcla_w['η_i_pura (Pa*s)'] = viscosidades_base_wilke
    
    col1_w, col2_w = st.columns([2, 1])
    with col1_w:
        st.subheader(f"📊 Composición y Viscosidades Puras ({modelo_seleccionado_wilke})")
        st.dataframe(df_mezcla_w.style.format({'X_i_raw': "{:.8f}", 'y_i (Normalizado)': "{:.4f}", 'η_i_pura (Pa*s)': "{:.4e}"}), use_container_width=True)
    
    y_array = df_mezcla_w['y_i (Normalizado)'].values
    eta_array_w = np.array(viscosidades_base_wilke)
    M_array = df_mezcla_w['M (g/mol)'].values
    
    viscosidad_mezcla_w, matriz_phi = regla_mezcla_wilke(y_array, eta_array_w, M_array)
    
    with col2_w:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.success("🏁 **Resultado Final (Wilke)**")
        st.metric(label="Viscosidad de la Mezcla (η_m)", value=f"{viscosidad_mezcla_w:.4e} Pa*s")
        st.info(f"Suma de X_i original: **{suma_X:.6f}**\n\n*(Se normalizó a 1.0 automáticamente)*")

# --- PESTAÑA MEZCLA HERNING-ZIPPERER (NUEVA) ---
with tab6:
    st.header("🧪 Viscosidad de la Mezcla (Herning-Zipperer)")
    st.markdown("Cálculo detallado de la regla de mezcla de Herning y Zipperer, dividiendo el cálculo en los parámetros intermedios del numerador y denominador.")
    
    modelo_seleccionado_hz = st.selectbox(
        "Elige el modelo para las viscosidades puras (η_i):",
        list(opciones_modelos.keys()),
        key="select_hz"
    )
    
    viscosidades_base_hz = opciones_modelos[modelo_seleccionado_hz]
    
    # 1 y 2. Construcción del DataFrame y asignación de viscosidad
    df_mezcla_hz = df_comp[['Componente', 'M (g/mol)', 'X_i_raw', 'y_i (Normalizado)']].copy()
    df_mezcla_hz['η_i_pura (Pa*s)'] = viscosidades_base_hz
    
    # 3. Cálculo de los parámetros intermedios (Numerador y Denominador parciales)
    df_mezcla_hz['X_i * √M_i'] = df_mezcla_hz['y_i (Normalizado)'] * np.sqrt(df_mezcla_hz['M (g/mol)'])
    df_mezcla_hz['X_i * η_i * √M_i'] = df_mezcla_hz['y_i (Normalizado)'] * df_mezcla_hz['η_i_pura (Pa*s)'] * np.sqrt(df_mezcla_hz['M (g/mol)'])
    
    # 4. Sumatorias
    suma_numerador = df_mezcla_hz['X_i * η_i * √M_i'].sum()
    suma_denominador = df_mezcla_hz['X_i * √M_i'].sum()
    
    # 5. Viscosidad total
    viscosidad_mezcla_hz = suma_numerador / suma_denominador
    
    col1_hz, col2_hz = st.columns([2, 1])
    
    with col1_hz:
        st.subheader("📊 Tabla de Parámetros Intermedios")
        st.dataframe(df_mezcla_hz.style.format({
            'X_i_raw': "{:.8f}", 
            'y_i (Normalizado)': "{:.6f}", 
            'η_i_pura (Pa*s)': "{:.4e}",
            'X_i * √M_i': "{:.4f}",
            'X_i * η_i * √M_i': "{:.4e}"
        }), use_container_width=True)
    
    with col2_hz:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.success("🏁 **Resultado Final (Herning-Zipperer)**")
        st.metric(label="Viscosidad de la Mezcla (η_m)", value=f"{viscosidad_mezcla_hz:.4e} Pa*s")
        st.divider()
        st.markdown("**Sumatorias Totales:**")
        st.write(f"**Numerador | Σ(X_i * η_i * √M_i):** {suma_numerador:.4e}")
        st.write(f"**Denominador | Σ(X_i * √M_i):** {suma_denominador:.4f}")
        st.info(f"Suma parcial (X_i original): **{suma_X:.8f}**\n\n*(La normalización dividió cada X_i entre {suma_X:.8f})*")