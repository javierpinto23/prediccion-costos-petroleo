import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import joblib

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predicción de Overrun Petrolero", layout="wide")

# 2. Cargar el modelo
@st.cache_resource
def load_model():
    # Asegúrate de que el archivo se llame exactamente así en GitHub
    return joblib.load('modelo_costos_catboost.pkl')

model = load_model()

st.title("📊 Predicción de Sobrecosto (Overrun) y Costo Final")
st.write("Introduzca los datos para calcular el factor de Overrun y el costo real estimado de la operación.")

# 3. Formulario de entrada de datos
with st.form("form_operacion"):
    st.subheader("📋 Información del Proyecto")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        proyecto = st.text_input("PROYECTO", value="Nombre del Proyecto")
        departamento = st.text_input("DEPARTAMENTO", value="CASANARE")
    with col_b:
        objetivo = st.text_input("OBJETIVO_EVENTO", value="PERFORACIÓN")
    with col_c:
        # Variable crítica para calcular el costo final después
        costo_planeado = st.number_input("COSTO_TOTAL_PLANEADO (US)", min_value=1.0, value=100000.0, step=1000.0)

    st.subheader("⚙️ Variables Técnicas (Códigos de Costo)")
    # Lista de tus variables numéricas exactas
    v_list = ["1000", "1200", "1300", "1400", "1500", "1600", "1700", "1800", 
              "1900", "2000", "2100", "2200", "2500", "2600", "2700", "2800", 
              "2900", "3000", "3100", "3200", "3300"]
    
    # Crear una cuadrícula de 7 columnas para que sea compacto
    tecnicas_input = {}
    cols = st.columns(7)
    for i, var in enumerate(v_list):
        with cols[i % 7]:
            tecnicas_input[var] = st.number_input(f"Cod {var}", value=0.0)

    submit = st.form_submit_button("CALCULAR PREDICCIÓN")

# 4. Lógica de cálculo
if submit:
    try:
        # Crear DataFrame con el orden exacto que espera el modelo
        datos_dict = {
            'PROYECTO': [proyecto],
            'DEPARTAMENTO': [departamento],
            'OBJETIVO_EVENTO': [objetivo]
        }
        # Agregar códigos numéricos
        for var in v_list:
            datos_dict[var] = [tecnicas_input[var]]
        
        # Agregar costo planeado
        datos_dict['COSTO_TOTAL_PLANEADO (US)'] = [costo_planeado]
        
        input_df = pd.DataFrame(datos_dict)

        # EL MODELO PREDICE EL OVERRUN
        overrun_predicho = model.predict(input_df)[0]
        
        # CÁLCULO DEL COSTO FINAL REAL
        # Si Overrun = Costo_Real / Costo_Planeado -> Costo_Real = Overrun * Costo_Planeado
        costo_final_estimado = overrun_predicho * costo_planeado

        # 5. Mostrar Resultados
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Factor de OVERRUN Predicho", value=f"{overrun_predicho:.4f}")
            st.info("Un Overrun > 1.0 indica que el costo final será superior al planeado.")

        with col_res2:
            st.metric(label="COSTO FINAL REAL ESTIMADO (US)", 
                      value=f"${costo_final_estimado:,.2f}",
                      delta=f"${costo_final_estimado - costo_planeado:,.2f} vs Planeado",
                      delta_color="inverse")

        # Alerta visual según el resultado
        if overrun_predicho > 1.10:
            st.error(f"⚠️ Alerta: Se predice un sobrecosto del {((overrun_predicho-1)*100):.1f}% sobre el presupuesto.")
        elif overrun_predicho < 0.95:
            st.success("✅ Eficiencia: El modelo predice un ahorro respecto al presupuesto planeado.")

    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
