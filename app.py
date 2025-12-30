import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Minería Miramar - Dashboard Inteligente", layout="wide")

# ESTILO PARA RECUADRO OSCURO Y LETRAS BLANCAS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    
    /* Cambiamos el fondo del recuadro a un Azul Oscuro Profesional */
    div[data-testid="stMetric"] {
        background-color: #1f77b4; /* Fondo azul fuerte */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
    }

    /* Aseguramos que el VALOR (el número) sea blanco puro */
    div[data-testid="stMetricValue"] > div {
        color: white !important;
        font-weight: bold;
    }

    /* Aseguramos que la ETIQUETA (el título) sea blanca con ligera transparencia */
    div[data-testid="stMetricLabel"] > div {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE MODELOS Y RECURSOS
@st.cache_resource
def load_resources():
    try:
        resources = {
            'lr': joblib.load(r'f:/Proyectos/Mineria de Datos/model_final_regression.pkl'),
            'imp_reg': joblib.load(r'f:/Proyectos/Mineria de Datos/imputer_regression.pkl'),
            'feats': joblib.load(r'f:/Proyectos/Mineria de Datos/features_list.pkl'),
            'iso': joblib.load(r'f:/Proyectos/Mineria de Datos/model_anomalias.pkl')
        }
        return resources
    except Exception as e:
        st.error(f"No se pudieron cargar todos los modelos. Asegúrate de ejecutar el script de entrenamiento primero. Error: {e}")
        return None

res = load_resources()

# 3. BARRA LATERAL (ENTRADA DE DATOS)
st.sidebar.header("📝 Perfil del Alumno")
st.sidebar.write("Ajusta los datos para simular predicciones:")

a1 = st.sidebar.slider("Asistencia Periodo 1 (%)", 0, 100, 85)
a2 = st.sidebar.slider("Asistencia Periodo 2 (%)", 0, 100, 85)
g1 = st.sidebar.number_input("Promedio Global P1", 1.0, 7.0, 5.0, step=0.1)

st.sidebar.subheader("Notas por Materia (P1)")
l1 = st.sidebar.slider("Lenguaje", 1.0, 7.0, 5.0)
m1 = st.sidebar.slider("Matemática", 1.0, 7.0, 5.0)
h1 = st.sidebar.slider("Historia", 1.0, 7.0, 5.0)
c1 = st.sidebar.slider("Ciencias", 1.0, 7.0, 5.0)
i1 = st.sidebar.slider("Inglés", 1.0, 7.0, 5.0)

# TÍTULO PRINCIPAL
st.title("📊 Sistema de Inteligencia Escolar - Colegio Miramar")
st.markdown("---")

if res:
    # 4. PESTAÑAS PRINCIPALES
    tab_pred, tab_impacto, tab_anomalias = st.tabs([
        "🔮 Predictor de Notas", 
        "🌳 Descubrimiento de Impacto", 
        "🚨 Detección de Anomalías"
    ])

    # --- PESTAÑA 1: PREDICTOR (REGRESIÓN LINEAL) ---
    with tab_pred:
        st.header("Predicción de Rendimiento Final")
        
        # Inferencia
        input_data = pd.DataFrame([[a1, a2, g1, l1, m1, h1, c1, i1]], columns=res['feats'])
        input_imp = res['imp_reg'].transform(input_data)
        prediccion = res['lr'].predict(input_imp)[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediccion,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Nota Final Estimada (P2)", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [1, 7], 'tickwidth': 1},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [1, 4], 'color': "#ff4b4b"},
                        {'range': [4, 5.5], 'color': "#f1c40f"},
                        {'range': [5.5, 7], 'color': "#2ecc71"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prediccion}}))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.metric("Estimación Numérica", f"{prediccion:.2f}")
            st.write("---")
            if prediccion < 4.5:
                st.error("⚠️ Alerta: El alumno proyecta un rendimiento en zona crítica.")
            elif prediccion >= 5.5:
                st.success("✨ El alumno proyecta un rendimiento de excelencia.")
            else:
                st.info("ℹ️ El alumno mantiene un rendimiento estable dentro del promedio.")

    # --- PESTAÑA 2: ÁRBOLES DE IMPACTO ---
    with tab_impacto:
        st.header("Descubrimiento de Patrones de Impacto")
        
        col_sel, col_desc = st.columns([1, 3])
        with col_sel:
            troncal = st.radio("Selecciona Asignatura Troncal:", ["Matematica", "Lenguaje", "Ciencias", "Historia"])
        
        with col_desc:
            img_path = f'arbol_impacto_{troncal}.png'
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Árbol de Decisión: Lógica de Riesgo en {troncal}", use_container_width=True)
                st.info(f"**Hallazgo clave:** Observa la raíz del árbol. Como descubrimos, el **Inglés (umbral 5.65)** es el principal filtro que separa el éxito del riesgo.")
            else:
                st.warning(f"No se encontró la imagen '{img_path}'.")

        st.markdown("---")
        st.subheader("📉 Punto de Quiebre de Asistencia")
        if os.path.exists('arbol_quiebre.png'):
            st.image('arbol_quiebre.png', caption="Análisis de Sensibilidad de Asistencia", use_container_width=True)
        

    # --- PESTAÑA 3: ANOMALÍAS (ISOLATION FOREST) ---
    with tab_anomalias:
        st.header("Casos Disonantes (Outliers)")
        st.write("Esta sección identifica alumnos que rompen la lógica común del colegio.")
        
        # 1. Carga de datos base
        df_csv = pd.read_csv('f:/Proyectos/Mineria de Datos/dataset_maestro_final_finalisimo.csv')
        df_anom = df_csv[['Asist_P2', 'Global_P2', 'RUT', 'Curso_Oficial']].dropna()
        
        # 2. FILTRO DE CURSOS (Dinámico)
        # Obtenemos la lista de cursos únicos para el selector
        lista_cursos = sorted(df_anom['Curso_Oficial'].unique().tolist())
        
        # Creamos el multiselect
        cursos_seleccionados = st.multiselect(
            "Filtrar por Curso(s):", 
            options=lista_cursos, 
            default=lista_cursos, # Por defecto todos seleccionados
            help="Selecciona uno, varios o quita todos para ver el panorama completo."
        )

        # Aplicamos el filtro al dataframe
        if cursos_seleccionados:
            df_filtrado = df_anom[df_anom['Curso_Oficial'].isin(cursos_seleccionados)].copy()
        else:
            st.warning("Selecciona al menos un curso para visualizar los datos.")
            df_filtrado = pd.DataFrame() # Vacío si no hay nada seleccionado

        if not df_filtrado.empty:
            # 3. Aplicamos el modelo guardado sobre los datos filtrados
            df_filtrado['Anomalia'] = res['iso'].predict(df_filtrado[['Asist_P2', 'Global_P2']])
            
            # 4. Visualización
            fig_anom = px.scatter(
                df_filtrado, 
                x="Asist_P2", 
                y="Global_P2", 
                color="Anomalia",
                hover_data=['RUT', 'Curso_Oficial'],
                color_discrete_map={-1: "red", 1: "lightgrey"},
                title=f"Mapa de Anomalías - Cursos: {', '.join(cursos_seleccionados[:3])}{'...' if len(cursos_seleccionados) > 3 else ''}"
            )
            st.plotly_chart(fig_anom, use_container_width=True)
            
            # Tabla de casos críticos detectados en los cursos seleccionados
            casos_criticos = df_filtrado[df_filtrado['Anomalia'] == -1]
            if not casos_criticos.empty:
                st.warning(f"⚠️ Se detectaron {len(casos_criticos)} casos disonantes en los cursos seleccionados:")
                st.dataframe(casos_criticos[['RUT', 'Curso_Oficial', 'Asist_P2', 'Global_P2']])
            else:
                st.success("✅ No se detectan anomalías en los cursos seleccionados.")

else:
    st.warning("Por favor, ejecuta el script de entrenamiento para generar los modelos necesarios.")