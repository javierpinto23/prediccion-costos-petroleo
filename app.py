import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import joblib

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predicción Costos Petroleros", layout="wide")

# 2. Cargar el modelo
# Nota: CatBoost se puede cargar con joblib o con su propio método model.load_model()
@st.cache_resource
def load_model():
    # Si lo guardaste con joblib:
    return joblib.load('modelo_costos_catboost.pkl')
    # Si lo guardaste con model.save_model(), usa: 
    # model = CatBoostRegressor()
    # return model.load_model('modelo_catboost.bin')

model = load_model()

st.title("🛢️ Predictor de Costos de Operación (CatBoost)")
st.write("Ajuste los parámetros técnicos y de ubicación para calcular el costo real esperado.")

# 3. Interfaz de usuario (Formulario)
with st.form("main_form"):
    st.subheader("Información General y Financiera")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        proyecto = st.text_input("PROYECTO", value="Proyecto Alpha")
    with c2:
        departamento = st.text_input("DEPARTAMENTO", value="HUILA")
    with c3:
        objetivo = st.text_input("OBJETIVO_EVENTO", value="PERFORACION")
    with c4:
        costo_planeado = st.number_input("COSTO_TOTAL_PLANEADO (US)", min_value=0.0, value=500000.0)

    st.subheader("Variables Técnicas / Códigos de Costos")
    # Creamos una cuadrícula para las variables numéricas para que sea fácil de llenar
    cols = st.columns(6)
    v_list = ["1000", "1200", "1300", "1400", "1500", "1600", "1700", "1800", 
              "1900", "2000", "2100", "2200", "2500", "2600", "2700", "2800", 
              "2900", "3000", "3100", "3200", "3300"]
    
    inputs_tecnicos = {}
    for i, var in enumerate(v_list):
        with cols[i % 6]:
            inputs_tecnicos[var] = st.number_input(f"Cod {var}", value=0.0)

    submit = st.form_submit_button("🚀 GENERAR PREDICCIÓN")

# 4. Lógica de Predicción
if submit:
    # Crear el diccionario con todas las variables en el orden exacto del modelo
    datos_dict = {
        'PROYECTO': [proyecto],
        'DEPARTAMENTO': [departamento],
        'OBJETIVO_EVENTO': [objetivo]
    }
    
    # Añadir las variables numéricas al diccionario
    for var in v_list:
        datos_dict[var] = [inputs_tecnicos[var]]
    
    # Añadir la última variable financiera
    datos_dict['COSTO_TOTAL_PLANEADO (US)'] = [costo_planeado]
    
    # Convertir a DataFrame
    input_df = pd.DataFrame(datos_dict)

    # Realizar la predicción
    try:
        resultado = model.predict(input_df)[0]*costo_planeado
        
        # Mostrar resultado con diseño llamativo
        st.success(f"### Estimación Final: ${resultado:,.2f} USD")
        
        # Comparación visual
        diferencia = resultado[0] - costo_planeado
        porcentaje = (diferencia / costo_planeado) * 100 if costo_planeado != 0 else 0
        
        st.metric("Desviación vs Planeado", 
                  value=f"${diferencia:,.2f}", 
                  delta=f"{porcentaje:.2f}%", 
                  delta_color="inverse")
    except Exception as e:

        st.error(f"Error en la predicción: {e}. Asegúrate de que las columnas coincidan con el entrenamiento.")
