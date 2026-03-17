import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador de Viscosidad", layout="wide")
st.title("🧪 Simulador de viscosidades de Coke Oven Gas & SoyBean Oil")

# ======================================= ===
# NAVEGACIÓN PRINCIPAL
# ==========================================
st.sidebar.title("🧭 Navegación")
seccion = st.sidebar.radio("Selecciona la Mezcla:", ["💨 Cake Oven Gas", "💧 SoyBean Oil"])
st.sidebar.markdown("---")

# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# 🟢 SECCIÓN 1: GASES (Tu código actual va aquí adentro)
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
if seccion == "💨 Cake Oven Gas":
    st.info("Cálculo de viscosidad mediante métodos del libro de Poling y literatura externa")

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

    def modelo_4_yoon_thodos(T, M, T_c, P_c_atm):
        # Corrección estricta con la literatura (Yoon-Thodos)
        P_c_pa = P_c_atm * 101325
        T_r = T / T_c
        xi = 2173.424 * (T_c ** (1/6)) * (M ** -0.5) * (P_c_pa ** -(2/3))
        termino = (46.10 * (T_r ** 0.618)) - (20.40 * np.exp(-0.449 * T_r)) + (19.40 * np.exp(-4.058 * T_r)) + 1
        eta_pa_s = (termino / xi) * 1e-8
        return eta_pa_s, T_r, xi

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

    def regla_mezcla_herning_zipperer_detallado(y_array, eta_array, M_array):
        n = len(y_array)
        xi_raiz_m = np.zeros(n)
        xi_eta_raiz_m = np.zeros(n)
        
        for i in range(n):
            xi_raiz_m[i] = y_array[i] * math.sqrt(M_array[i])
            xi_eta_raiz_m[i] = y_array[i] * eta_array[i] * math.sqrt(M_array[i])
            
        suma_num = np.sum(xi_eta_raiz_m)
        suma_den = np.sum(xi_raiz_m)
        eta_mezcla = suma_num / suma_den
        return eta_mezcla, xi_raiz_m, xi_eta_raiz_m, suma_num, suma_den

    def generar_curva_viscosidad_T(modelo_nombre, y_array, M_array, df, regla='wilke'):
        T_rango = np.linspace(250, 600, 30)
        eta_mezcla_rango = []
        
        for t in T_rango:
            eta_i_t = []
            for idx, row in df.iterrows():
                if "1" in modelo_nombre:
                    v, _, _ = modelo_1_chapman_enskog(t, row['M (g/mol)'], row['σ (Å)'], row['ε/k (K)'])
                elif "2" in modelo_nombre:
                    v, _, _ = modelo_2_estados_correspondientes(t, row['M (g/mol)'], row['V_c (cm3/mol)'], row['T_c (K)'])
                elif "3" in modelo_nombre:
                    v = modelo_3_sutherland(t, row['η_0 (Pa*s)'], row['T_0 (K)'], row['S (K)'])
                elif "4" in modelo_nombre:
                    v, _, _ = modelo_4_yoon_thodos(t, row['M (g/mol)'], row['T_c (K)'], row['P_c (atm)'])
                eta_i_t.append(v)
                
            eta_array_t = np.array(eta_i_t)
            
            if regla == 'wilke':
                eta_m, _ = regla_mezcla_wilke(y_array, eta_array_t, M_array)
            elif regla == 'hz':
                eta_m, _, _, _, _ = regla_mezcla_herning_zipperer_detallado(y_array, eta_array_t, M_array)
                
            eta_mezcla_rango.append(eta_m)
            
        return T_rango, eta_mezcla_rango


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
        visc_m4, T_r_m4, xi = modelo_4_yoon_thodos(t_val, row['M (g/mol)'], row['T_c (K)'], row['P_c (atm)'])
        resultados_m4_global.append({'Componente': comp, 'T (K)': t_val, 'T_r': round(T_r_m4, 4), 'ξ (Xi)': round(xi, 6), 'Viscosidad Calc (Pa*s)': visc_m4})
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
        st.subheader("📋 Tabla de Resultados Calculados")
        st.dataframe(df_res_m4.style.format({'Viscosidad Calc (Pa*s)': "{:.4e}", 'ξ (Xi)': "{:.6e}"}), use_container_width=True)

        st.divider()
        st.subheader("📊 Comparación con Datos Experimentales")
        df_comparacion_m4 = pd.merge(df_exp, df_res_m4, on='Componente')
        df_comparacion_m4['% Error'] = abs(df_comparacion_m4['Viscosidad Calc (Pa*s)'] - df_comparacion_m4['Viscosidad Exp (Pa*s)']) / df_comparacion_m4['Viscosidad Exp (Pa*s)'] * 100
        st.dataframe(df_comparacion_m4[['Componente', 'T (K)', 'Viscosidad Exp (Pa*s)', 'Viscosidad Calc (Pa*s)', '% Error']].style.format({'Viscosidad Exp (Pa*s)': "{:.4e}", 'Viscosidad Calc (Pa*s)': "{:.4e}", '% Error': "{:.2f} %"}), use_container_width=True)

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

            # GRÁFICA WILKE
        st.divider()
        st.subheader(f"📈 Comportamiento Térmico de la Mezcla ({modelo_seleccionado_wilke} + Wilke)")
        T_plot_w, eta_plot_w = generar_curva_viscosidad_T(modelo_seleccionado_wilke, y_array, M_array, df_comp, regla='wilke')
        
        fig_w, ax_w = plt.subplots(figsize=(10, 4))
        ax_w.plot(T_plot_w, eta_plot_w, color='crimson', linewidth=2.5, marker='o', markersize=4)
        ax_w.set_xlabel('Temperatura (K)', fontweight='bold')
        ax_w.set_ylabel('Viscosidad de Mezcla (Pa*s)', fontweight='bold')
        ax_w.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_w)


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

            # GRÁFICA HERNING-ZIPPERER
        st.divider()
        st.subheader(f"📈 Comportamiento Térmico de la Mezcla ({modelo_seleccionado_hz} + Herning-Zipperer)")
        T_plot_hz, eta_plot_hz = generar_curva_viscosidad_T(modelo_seleccionado_hz, y_array, M_array, df_comp, regla='hz')
        
        fig_hz, ax_hz = plt.subplots(figsize=(10, 4))
        ax_hz.plot(T_plot_hz, eta_plot_hz, color='darkcyan', linewidth=2.5, marker='s', markersize=4)
        ax_hz.set_xlabel('Temperatura (K)', fontweight='bold')
        ax_hz.set_ylabel('Viscosidad de Mezcla (Pa*s)', fontweight='bold')
        ax_hz.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_hz)


# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# 🔵 SECCIÓN 2: LÍQUIDOS
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
elif seccion == "💧 SoyBean Oil":
    st.header("💧 Módulo de Viscosidad para Líquidos")
    st.info("Cálculo de viscosidad mediante métodos del libro de Poling y literatura externa")
    
    # 1. BASE DE DATOS DE LÍQUIDOS
    datos_liquidos = {
        'Componente': ["Ácido Oleico", "Ácido Linoleico", "Ácido Linolénico", "Ácido Palmítico", "Ácido Esteárico"],
        'M (g/mol)': [282.46, 280.45, 278.43, 256.43, 284.48],
        'Densidad (g/cm3)': [0.895, 0.902, 0.914, 0.852, 0.847],
        'E_act (J/mol)': [25000, 24000, 23000, 27000, 28000],
        'C_totales': [18, 18, 18, 16, 18],
        'Dobles_Enlaces': [1, 2, 3, 0, 0],
        'CH3': [1, 1, 1, 1, 1],
        'CH2': [14, 12, 10, 13, 15],
        'COOH': [1, 1, 1, 1, 1],
        'T_b (K)': [633.15, 638.15, 638.15, 624.6, 649.25],
        'T_exp_default (K)': [353.15, 353.15, 353.15, 353.15, 353.15],
        'Peso (g)': [18.93, 53.67, 6.28, 15.64, 3.74] # Datos de tu Excel
    }
    df_liq = pd.DataFrame(datos_liquidos)
    
    # Cálculo de Moles y Fracción Molar (x_i)
    df_liq['Moles'] = df_liq['Peso (g)'] / df_liq['M (g/mol)']
    moles_totales = df_liq['Moles'].sum()
    df_liq['x_i'] = df_liq['Moles'] / moles_totales

    # 2. FUNCIONES DE LÍQUIDOS
    def modelo_L1_sastri_rao(T, Tb, n_c, n_db, n_ch3, n_ch2, n_cooh):
        sum_delta_nb = (n_ch3 * 0.105) + (n_ch2 * 0.0) + (n_db * -0.005) + (n_cooh * 0.250)
        n_sastri = 0.2 + (n_cooh * 0.100) + (0.050 if n_c > 8 else 0.0)
        tr = (T / Tb)
        f_termico = (3 - 2 * tr)**0.19
        term_a = 1 - (f_termico / tr)
        term_b = 0.38 * f_termico * math.log(tr)
        ln_pvp = (4.5398 + 1.0309 * math.log(Tb)) * (term_a - term_b)
        pvp = math.exp(ln_pvp)
        visc_sastri = sum_delta_nb * (pvp ** (-n_sastri))
        return visc_sastri
    
    def modelo_L2_orrick_erbar(T, M, rho, n_c, n_db, n_cooh):
        n_orrick = n_c - n_cooh
        a_oe = -(6.95 + 0.21 * n_orrick) + (n_db * 0.24) + (n_cooh * -0.90)
        b_oe = (275 + 99 * n_orrick) + (n_db * -90) + (n_cooh * 770)
        visc_orrick = (rho * M) * math.exp(a_oe + (b_oe / T))
        return visc_orrick

    def modelo_L3_van_velzen(T, n_c, n_db, n_cooh):
        n_ast = n_c + (n_db * (-0.152 - (0.042 * n_c))) + (n_cooh * 10.71)
        b0 = 530.59 + 13.74 * n_ast
        t0 = 238.59 + 8.164 * n_ast
        b_vv = b0 + (n_db * (-44.94 + 5.41 * n_ast)) + (n_cooh * (-249.12 + 22.449 * n_ast))
        visc_van_velzen = 10 ** (b_vv * ((1 / T) - (1 / t0)))
        return visc_van_velzen

    def modelo_L4_eyring(T, M, rho, E_act):
        h, na, r = 6.63e-34, 6.02e23, 8.314
        vol_molar = (M / 1000) / (rho * 1000)
        visc_eyring = ((h * na / vol_molar) * math.exp(E_act / (r * T))) * 1000
        return visc_eyring

    # ==========================================
    # 3. INTERFAZ: PANEL LATERAL DE LÍQUIDOS
    # ==========================================
    st.sidebar.header("⚙️ Condiciones de Operación (Líquidos)")
    st.sidebar.markdown("Ajusta Temperatura y Viscosidad Experimental para cada ácido:")

    resultados_L1, resultados_L2, resultados_L3, resultados_L4 = [], [], [], []
    viscosidades_exp = []

    # Valores por defecto
    v_exp_defaults = [5.60, 4.57, 6.90, 5.80, 7.31] 

    for index, row in df_liq.iterrows():
        comp = row['Componente']
        st.sidebar.markdown(f"**{comp}**")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            t_val = st.sidebar.number_input("T (K)", value=float(row['T_exp_default (K)']), step=1.0, key=f"t_liq_{comp}")
        with col2:
            v_val = st.sidebar.number_input("Visc Exp (cP)", value=v_exp_defaults[index], step=0.1, key=f"v_exp_{comp}")
            
        viscosidades_exp.append(v_val)
        
        # Cálculos
        v_l1 = modelo_L1_sastri_rao(t_val, row['T_b (K)'], row['C_totales'], row['Dobles_Enlaces'], row['CH3'], row['CH2'], row['COOH'])
        v_l2 = modelo_L2_orrick_erbar(t_val, row['M (g/mol)'], row['Densidad (g/cm3)'], row['C_totales'], row['Dobles_Enlaces'], row['COOH'])
        v_l3 = modelo_L3_van_velzen(t_val, row['C_totales'], row['Dobles_Enlaces'], row['COOH'])
        v_l4 = modelo_L4_eyring(t_val, row['M (g/mol)'], row['Densidad (g/cm3)'], row['E_act (J/mol)'])
        
        # Guardar resultados
        resultados_L1.append({'Componente': comp, 'T_operacion (K)': t_val, 'T_ebullicion (K)': row['T_b (K)'], 'Viscosidad Calc (cP)': v_l1})
        resultados_L2.append({'Componente': comp, 'T_operacion (K)': t_val, 'Viscosidad Calc (cP)': v_l2})
        resultados_L3.append({'Componente': comp, 'T_operacion (K)': t_val, 'Viscosidad Calc (cP)': v_l3})
        resultados_L4.append({'Componente': comp, 'T_operacion (K)': t_val, 'E_Activación (J/mol)': row['E_act (J/mol)'], 'Viscosidad Calc (cP)': v_l4})
        
        st.sidebar.markdown("---")
        
    # ==========================================
    # 4. PESTAÑAS Y TABLAS PARA LÍQUIDOS (¡FUERA DEL FOR LOOP!)
    # ==========================================
    tab_L1, tab_L2, tab_L3, tab_L4, tab_L5, tab_L6 = st.tabs([
        "🔴 L1: Sastri-Rao", "🟠 L2: Orrick-Erbar", "🟡 L3: Van Velzen", "🟢 L4: Eyring", "🔵 L5: GRUNBERG AND NISSAN", "🟣 L6: KENDALL MONROE"
    ])

    # --- PESTAÑA L1: SASTRI-RAO ---
    with tab_L1:
        st.subheader("📋 Resultados Método Sastri-Rao")
        df_comp_L1 = pd.DataFrame(resultados_L1)
        df_comp_L1['Viscosidad Exp (cP)'] = viscosidades_exp
        df_comp_L1['% Error'] = abs(df_comp_L1['Viscosidad Exp (cP)'] - df_comp_L1['Viscosidad Calc (cP)']) / df_comp_L1['Viscosidad Exp (cP)'] * 100
        
        st.dataframe(df_comp_L1[['Componente', 'T_operacion (K)', 'Viscosidad Exp (cP)', 'Viscosidad Calc (cP)', '% Error']].style.format({
            'T_operacion (K)': "{:.2f}", 'Viscosidad Exp (cP)': "{:.4f}", 'Viscosidad Calc (cP)': "{:.4f}", '% Error': "{:.2f} %"
        }), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Comparación con Datos Experimentales")
        
        y_true_L1, y_pred_L1 = df_comp_L1['Viscosidad Exp (cP)'], df_comp_L1['Viscosidad Calc (cP)']
        ss_res_L1, ss_tot_L1 = np.sum((y_true_L1 - y_pred_L1)**2), np.sum((y_true_L1 - np.mean(y_true_L1))**2)
        r2_L1 = 1 - (ss_res_L1 / ss_tot_L1) if ss_tot_L1 != 0 else 0
        
        col_grafica_L1, col_metricas_L1 = st.columns([3, 1])
        with col_grafica_L1:
            fig_L1, ax_L1 = plt.subplots(figsize=(7, 5))
            ax_L1.scatter(y_true_L1, y_pred_L1, color='crimson', edgecolor='black', s=80, label='Predicción L1', zorder=3)
            min_v, max_v = min(y_true_L1.min(), y_pred_L1.min())*0.95, max(y_true_L1.max(), y_pred_L1.max())*1.05
            ax_L1.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal', zorder=2)
            ax_L1.set(xlabel='Viscosidad Experimental (cP)', ylabel='Viscosidad Calculada (cP)')
            ax_L1.grid(True, linestyle=':', alpha=0.7)
            ax_L1.legend()
            st.pyplot(fig_L1)
        with col_metricas_L1:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric(label="Coeficiente R²", value=f"{r2_L1:.4f}")
            st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comp_L1['% Error']):.2f} %")

    # --- PESTAÑA L2: ORRICK & ERBAR ---
    with tab_L2:
        st.subheader("📋 Resultados Método Orrick & Erbar")
        df_comp_L2 = pd.DataFrame(resultados_L2)
        df_comp_L2['Viscosidad Exp (cP)'] = viscosidades_exp
        df_comp_L2['% Error'] = abs(df_comp_L2['Viscosidad Exp (cP)'] - df_comp_L2['Viscosidad Calc (cP)']) / df_comp_L2['Viscosidad Exp (cP)'] * 100
        
        st.dataframe(df_comp_L2[['Componente', 'T_operacion (K)', 'Viscosidad Exp (cP)', 'Viscosidad Calc (cP)', '% Error']].style.format({
            'T_operacion (K)': "{:.2f}", 'Viscosidad Exp (cP)': "{:.4f}", 'Viscosidad Calc (cP)': "{:.4f}", '% Error': "{:.2f} %"
        }), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Comparación con Datos Experimentales")
        
        y_true_L2, y_pred_L2 = df_comp_L2['Viscosidad Exp (cP)'], df_comp_L2['Viscosidad Calc (cP)']
        ss_res_L2, ss_tot_L2 = np.sum((y_true_L2 - y_pred_L2)**2), np.sum((y_true_L2 - np.mean(y_true_L2))**2)
        r2_L2 = 1 - (ss_res_L2 / ss_tot_L2) if ss_tot_L2 != 0 else 0
        
        col_grafica_L2, col_metricas_L2 = st.columns([3, 1])
        with col_grafica_L2:
            fig_L2, ax_L2 = plt.subplots(figsize=(7, 5))
            ax_L2.scatter(y_true_L2, y_pred_L2, color='darkorange', edgecolor='black', s=80, label='Predicción L2', zorder=3)
            min_v, max_v = min(y_true_L2.min(), y_pred_L2.min())*0.95, max(y_true_L2.max(), y_pred_L2.max())*1.05
            ax_L2.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal', zorder=2)
            ax_L2.set(xlabel='Viscosidad Experimental (cP)', ylabel='Viscosidad Calculada (cP)')
            ax_L2.grid(True, linestyle=':', alpha=0.7)
            ax_L2.legend()
            st.pyplot(fig_L2)
        with col_metricas_L2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric(label="Coeficiente R²", value=f"{r2_L2:.4f}")
            st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comp_L2['% Error']):.2f} %")

    # --- PESTAÑA L3: VAN VELZEN ---
    with tab_L3:
        st.subheader("📋 Resultados Método Van Velzen")
        df_comp_L3 = pd.DataFrame(resultados_L3)
        df_comp_L3['Viscosidad Exp (cP)'] = viscosidades_exp
        df_comp_L3['% Error'] = abs(df_comp_L3['Viscosidad Exp (cP)'] - df_comp_L3['Viscosidad Calc (cP)']) / df_comp_L3['Viscosidad Exp (cP)'] * 100
        
        st.dataframe(df_comp_L3[['Componente', 'T_operacion (K)', 'Viscosidad Exp (cP)', 'Viscosidad Calc (cP)', '% Error']].style.format({
            'T_operacion (K)': "{:.2f}", 'Viscosidad Exp (cP)': "{:.4f}", 'Viscosidad Calc (cP)': "{:.4f}", '% Error': "{:.2f} %"
        }), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Comparación con Datos Experimentales")
        
        y_true_L3, y_pred_L3 = df_comp_L3['Viscosidad Exp (cP)'], df_comp_L3['Viscosidad Calc (cP)']
        ss_res_L3, ss_tot_L3 = np.sum((y_true_L3 - y_pred_L3)**2), np.sum((y_true_L3 - np.mean(y_true_L3))**2)
        r2_L3 = 1 - (ss_res_L3 / ss_tot_L3) if ss_tot_L3 != 0 else 0
        
        col_grafica_L3, col_metricas_L3 = st.columns([3, 1])
        with col_grafica_L3:
            fig_L3, ax_L3 = plt.subplots(figsize=(7, 5))
            ax_L3.scatter(y_true_L3, y_pred_L3, color='teal', edgecolor='black', s=80, label='Predicción L3', zorder=3)
            min_v, max_v = min(y_true_L3.min(), y_pred_L3.min())*0.95, max(y_true_L3.max(), y_pred_L3.max())*1.05
            ax_L3.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal', zorder=2)
            ax_L3.set(xlabel='Viscosidad Experimental (cP)', ylabel='Viscosidad Calculada (cP)')
            ax_L3.grid(True, linestyle=':', alpha=0.7)
            ax_L3.legend()
            st.pyplot(fig_L3)
        with col_metricas_L3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric(label="Coeficiente R²", value=f"{r2_L3:.4f}")
            st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comp_L3['% Error']):.2f} %")

    # --- PESTAÑA L4: EYRING ---
    with tab_L4:
        st.subheader("📋 Resultados Ecuación de Eyring")
        df_comp_L4 = pd.DataFrame(resultados_L4)
        df_comp_L4['Viscosidad Exp (cP)'] = viscosidades_exp
        df_comp_L4['% Error'] = abs(df_comp_L4['Viscosidad Exp (cP)'] - df_comp_L4['Viscosidad Calc (cP)']) / df_comp_L4['Viscosidad Exp (cP)'] * 100
        
        st.dataframe(df_comp_L4[['Componente', 'T_operacion (K)', 'E_Activación (J/mol)', 'Viscosidad Exp (cP)', 'Viscosidad Calc (cP)', '% Error']].style.format({
            'T_operacion (K)': "{:.2f}", 'Viscosidad Exp (cP)': "{:.4f}", 'Viscosidad Calc (cP)': "{:.4f}", '% Error': "{:.2f} %"
        }), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Comparación con Datos Experimentales")
        
        y_true_L4, y_pred_L4 = df_comp_L4['Viscosidad Exp (cP)'], df_comp_L4['Viscosidad Calc (cP)']
        ss_res_L4, ss_tot_L4 = np.sum((y_true_L4 - y_pred_L4)**2), np.sum((y_true_L4 - np.mean(y_true_L4))**2)
        r2_L4 = 1 - (ss_res_L4 / ss_tot_L4) if ss_tot_L4 != 0 else 0
        
        col_grafica_L4, col_metricas_L4 = st.columns([3, 1])
        with col_grafica_L4:
            fig_L4, ax_L4 = plt.subplots(figsize=(7, 5))
            ax_L4.scatter(y_true_L4, y_pred_L4, color='forestgreen', edgecolor='black', s=80, label='Predicción L4', zorder=3)
            min_v, max_v = min(y_true_L4.min(), y_pred_L4.min())*0.95, max(y_true_L4.max(), y_pred_L4.max())*1.05
            ax_L4.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal', zorder=2)
            ax_L4.set(xlabel='Viscosidad Experimental (cP)', ylabel='Viscosidad Calculada (cP)')
            ax_L4.grid(True, linestyle=':', alpha=0.7)
            ax_L4.legend()
            st.pyplot(fig_L4)
        with col_metricas_L4:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric(label="Coeficiente R²", value=f"{r2_L4:.4f}")
            st.metric(label="Error Global (MAPE)", value=f"{np.mean(df_comp_L4['% Error']):.2f} %")
            
# --- PESTAÑA L5: MEZCLA (GRUNBERG & NISSAN) ---
    with tab_L5:
        st.subheader("🛢️ Método 5: Mezcla con Grunberg y Nissan (1949)")
        st.markdown("**Regla de Mezclado:** Asumiendo parámetro de interacción $G_{ij} = 0$")
        st.latex(r"\ln \eta_m = \sum x_i \ln \eta_i \implies \eta_m = \exp \left( \sum x_i \ln \eta_i \right)")
        
        # Input exclusivo para la mezcla (actualiza la gráfica y la tabla dinámicamente)
        T_mezcla = st.number_input("🌡️ Temperatura de la Mezcla (K)", value=353.15, step=1.0, key="t_mezcla_grunberg")
        
        # 1. CÁLCULO DEL PUNTO ESPECÍFICO (Tabla)
        x_i_array = df_liq['x_i'].values
        v1_mix, v2_mix, v3_mix, v4_mix = [], [], [], []
        
        # Calculamos la viscosidad de cada componente a la T_mezcla elegida
        for index, row in df_liq.iterrows():
            v1_mix.append(modelo_L1_sastri_rao(T_mezcla, row['T_b (K)'], row['C_totales'], row['Dobles_Enlaces'], row['CH3'], row['CH2'], row['COOH']))
            v2_mix.append(modelo_L2_orrick_erbar(T_mezcla, row['M (g/mol)'], row['Densidad (g/cm3)'], row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
            v3_mix.append(modelo_L3_van_velzen(T_mezcla, row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
            v4_mix.append(modelo_L4_eyring(T_mezcla, row['M (g/mol)'], row['Densidad (g/cm3)'], row['E_act (J/mol)']))
            
        # Aplicamos Grunberg-Nissan
        mezcla_M1 = np.exp(np.sum(x_i_array * np.log(v1_mix)))
        mezcla_M2 = np.exp(np.sum(x_i_array * np.log(v2_mix)))
        mezcla_M3 = np.exp(np.sum(x_i_array * np.log(v3_mix)))
        mezcla_M4 = np.exp(np.sum(x_i_array * np.log(v4_mix)))
        
        df_mezcla = pd.DataFrame({
            'Método Base': ['M1 (Sastri-Rao)', 'M2 (Orrick-Erbar)', 'M3 (Van Velzen)', 'M4 (Eyring)'],
            'Temperatura (K)': [T_mezcla]*4,
            'Viscosidad Mezcla (cP)': [mezcla_M1, mezcla_M2, mezcla_M3, mezcla_M4]
        })
        
        col_tabla, col_grafica = st.columns([1, 1.5])
        
        with col_tabla:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_mezcla.style.format({
                'Temperatura (K)': "{:.2f}",
                'Viscosidad Mezcla (cP)': "{:.4f}"
            }), use_container_width=True)
            
        with col_grafica:
            # 2. GENERACIÓN DE LA CURVA (Rango de -40K a +40K alrededor de la T elegida)
            T_range = np.linspace(T_mezcla - 40, T_mezcla + 40, 30)
            mix_curve_m1, mix_curve_m2, mix_curve_m3, mix_curve_m4 = [], [], [], []
            
            # Calculamos toda la curva de viscosidad para la gráfica
            for t in T_range:
                v1_t, v2_t, v3_t, v4_t = [], [], [], []
                for idx, row in df_liq.iterrows():
                    v1_t.append(modelo_L1_sastri_rao(t, row['T_b (K)'], row['C_totales'], row['Dobles_Enlaces'], row['CH3'], row['CH2'], row['COOH']))
                    v2_t.append(modelo_L2_orrick_erbar(t, row['M (g/mol)'], row['Densidad (g/cm3)'], row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
                    v3_t.append(modelo_L3_van_velzen(t, row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
                    v4_t.append(modelo_L4_eyring(t, row['M (g/mol)'], row['Densidad (g/cm3)'], row['E_act (J/mol)']))
                
                mix_curve_m1.append(np.exp(np.sum(x_i_array * np.log(v1_t))))
                mix_curve_m2.append(np.exp(np.sum(x_i_array * np.log(v2_t))))
                mix_curve_m3.append(np.exp(np.sum(x_i_array * np.log(v3_t))))
                mix_curve_m4.append(np.exp(np.sum(x_i_array * np.log(v4_t))))
                
            fig_mix, ax_mix = plt.subplots(figsize=(7, 5))
            
            # Dibujar las curvas de tendencia
            ax_mix.plot(T_range, mix_curve_m1, color='crimson', label='M1', linewidth=2)
            ax_mix.plot(T_range, mix_curve_m2, color='darkorange', label='M2', linewidth=2)
            ax_mix.plot(T_range, mix_curve_m3, color='teal', label='M3', linewidth=2)
            ax_mix.plot(T_range, mix_curve_m4, color='forestgreen', label='M4', linewidth=2)
            
            # Dibujar los puntos exactos a la Temperatura elegida
            ax_mix.scatter([T_mezcla]*4, [mezcla_M1, mezcla_M2, mezcla_M3, mezcla_M4], 
                           color=['crimson', 'darkorange', 'teal', 'forestgreen'], s=100, zorder=5, edgecolor='black')
                           
            ax_mix.set_xlabel('Temperatura (K)', fontweight='bold')
            ax_mix.set_ylabel('Viscosidad de Mezcla (cP)', fontweight='bold')
            ax_mix.set_title('Comportamiento de la Mezcla vs Temperatura', fontweight='bold')
            ax_mix.grid(True, linestyle=':', alpha=0.7)
            ax_mix.legend()
            
            st.pyplot(fig_mix)


# --- PESTAÑA L6: MEZCLA (KENDALL MONROE) ---
    with tab_L6:
        st.subheader("🛢️ Método 6: Mezcla con Kendall y Monroe (1917)")
        st.markdown("**Regla de Mezclado:** Basada en la raíz cúbica de las viscosidades puras")
        st.latex(r"\eta_m^{1/3} = \sum x_i \eta_i^{1/3} \implies \eta_m = \left( \sum x_i \eta_i^{1/3} \right)^3")
        
        # Input exclusivo para la mezcla limitado a rango de líquidos
        T_mezcla_km = st.number_input(
            "🌡️ Temperatura de la Mezcla (K)", 
            min_value=273.15, 
            max_value=580.0, 
            value=353.15, 
            step=1.0, 
            key="t_mezcla_kendall"
        )
        
        # 1. CÁLCULO DEL PUNTO ESPECÍFICO (Tabla)
        x_i_array = df_liq['x_i'].values
        v1_mix_km, v2_mix_km, v3_mix_km, v4_mix_km = [], [], [], []
        
        # Calculamos la viscosidad de cada componente a la T_mezcla elegida
        for index, row in df_liq.iterrows():
            v1_mix_km.append(modelo_L1_sastri_rao(T_mezcla_km, row['T_b (K)'], row['C_totales'], row['Dobles_Enlaces'], row['CH3'], row['CH2'], row['COOH']))
            v2_mix_km.append(modelo_L2_orrick_erbar(T_mezcla_km, row['M (g/mol)'], row['Densidad (g/cm3)'], row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
            v3_mix_km.append(modelo_L3_van_velzen(T_mezcla_km, row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
            v4_mix_km.append(modelo_L4_eyring(T_mezcla_km, row['M (g/mol)'], row['Densidad (g/cm3)'], row['E_act (J/mol)']))
            
        # Convertir a arrays de numpy para hacer las operaciones de raíz cúbica
        v1_mix_km = np.array(v1_mix_km)
        v2_mix_km = np.array(v2_mix_km)
        v3_mix_km = np.array(v3_mix_km)
        v4_mix_km = np.array(v4_mix_km)
            
        # Aplicamos Kendall-Monroe: Sumatoria de (xi * visc^(1/3)) y todo eso elevado al cubo
        mezcla_M1_km = (np.sum(x_i_array * (v1_mix_km ** (1/3)))) ** 3
        mezcla_M2_km = (np.sum(x_i_array * (v2_mix_km ** (1/3)))) ** 3
        mezcla_M3_km = (np.sum(x_i_array * (v3_mix_km ** (1/3)))) ** 3
        mezcla_M4_km = (np.sum(x_i_array * (v4_mix_km ** (1/3)))) ** 3
        
        df_mezcla_km = pd.DataFrame({
            'Método Base': ['M1 (Sastri-Rao)', 'M2 (Orrick-Erbar)', 'M3 (Van Velzen)', 'M4 (Eyring)'],
            'Temperatura (K)': [T_mezcla_km]*4,
            'Viscosidad Mezcla (cP)': [mezcla_M1_km, mezcla_M2_km, mezcla_M3_km, mezcla_M4_km]
        })
        
        col_tabla_km, col_grafica_km = st.columns([1, 1.5])
        
        with col_tabla_km:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_mezcla_km.style.format({
                'Temperatura (K)': "{:.2f}",
                'Viscosidad Mezcla (cP)': "{:.4f}"
            }), use_container_width=True)
            
        with col_grafica_km:
            # 2. GENERACIÓN DE LA CURVA (Rango de -40K a +40K alrededor de la T elegida)
            T_range_km = np.linspace(T_mezcla_km - 40, T_mezcla_km + 40, 30)
            mix_curve_m1_km, mix_curve_m2_km, mix_curve_m3_km, mix_curve_m4_km = [], [], [], []
            
            # Calculamos toda la curva de viscosidad para la gráfica
            for t in T_range_km:
                v1_t, v2_t, v3_t, v4_t = [], [], [], []
                for idx, row in df_liq.iterrows():
                    v1_t.append(modelo_L1_sastri_rao(t, row['T_b (K)'], row['C_totales'], row['Dobles_Enlaces'], row['CH3'], row['CH2'], row['COOH']))
                    v2_t.append(modelo_L2_orrick_erbar(t, row['M (g/mol)'], row['Densidad (g/cm3)'], row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
                    v3_t.append(modelo_L3_van_velzen(t, row['C_totales'], row['Dobles_Enlaces'], row['COOH']))
                    v4_t.append(modelo_L4_eyring(t, row['M (g/mol)'], row['Densidad (g/cm3)'], row['E_act (J/mol)']))
                
                v1_t = np.array(v1_t)
                v2_t = np.array(v2_t)
                v3_t = np.array(v3_t)
                v4_t = np.array(v4_t)
                
                mix_curve_m1_km.append((np.sum(x_i_array * (v1_t ** (1/3)))) ** 3)
                mix_curve_m2_km.append((np.sum(x_i_array * (v2_t ** (1/3)))) ** 3)
                mix_curve_m3_km.append((np.sum(x_i_array * (v3_t ** (1/3)))) ** 3)
                mix_curve_m4_km.append((np.sum(x_i_array * (v4_t ** (1/3)))) ** 3)
                
            fig_mix_km, ax_mix_km = plt.subplots(figsize=(7, 5))
            
            # Dibujar las curvas de tendencia
            ax_mix_km.plot(T_range_km, mix_curve_m1_km, color='crimson', label='M1', linewidth=2)
            ax_mix_km.plot(T_range_km, mix_curve_m2_km, color='darkorange', label='M2', linewidth=2)
            ax_mix_km.plot(T_range_km, mix_curve_m3_km, color='teal', label='M3', linewidth=2)
            ax_mix_km.plot(T_range_km, mix_curve_m4_km, color='forestgreen', label='M4', linewidth=2)
            
            # Dibujar los puntos exactos a la Temperatura elegida
            ax_mix_km.scatter([T_mezcla_km]*4, [mezcla_M1_km, mezcla_M2_km, mezcla_M3_km, mezcla_M4_km], 
                           color=['crimson', 'darkorange', 'teal', 'forestgreen'], s=100, zorder=5, edgecolor='black')
                           
            ax_mix_km.set_xlabel('Temperatura (K)', fontweight='bold')
            ax_mix_km.set_ylabel('Viscosidad de Mezcla (cP)', fontweight='bold')
            ax_mix_km.set_title('Kendall-Monroe: Mezcla vs Temperatura', fontweight='bold')
            ax_mix_km.grid(True, linestyle=':', alpha=0.7)
            ax_mix_km.legend()
            
            st.pyplot(fig_mix_km)