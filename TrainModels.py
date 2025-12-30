import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
# Cambia esta ruta si es necesario
PATH_CSV = r'f:/Proyectos/Mineria de Datos/dataset_maestro_final_finalisimo.csv'

if not os.path.exists(PATH_CSV):
    print(f"❌ ERROR: No se encontró el archivo en {PATH_CSV}")
    exit()

print("🚀 Cargando datos y comenzando proceso de Minería...")
df = pd.read_csv(PATH_CSV)

# ==========================================
# 2. MODELO PREDICTIVO: REGRESIÓN LINEAL
# ==========================================
print("🔮 Entrenando Modelo Predictivo (Regresión Lineal)...")
features_reg = [
    'Asist_P1', 'Asist_P2', 'Global_P1', 
    'Lenguaje_P1', 'Matematica_P1', 'Historia_P1', 'Ciencias_P1', 'Ingles_P1'
]
target_reg = 'Global_P2'

# Limpieza y preparación
df_reg = df.dropna(subset=[target_reg]).copy()
X_reg = df_reg[features_reg]
y_reg = df_reg[target_reg]

imputer_reg = SimpleImputer(strategy='median')
X_reg_imp = imputer_reg.fit_transform(X_reg)

model_lr = LinearRegression()
model_lr.fit(X_reg_imp, y_reg)

# Guardar predictor
joblib.dump(model_lr, 'model_final_regression.pkl')
joblib.dump(imputer_reg, 'imputer_regression.pkl')
joblib.dump(features_reg, 'features_list.pkl')

# ==========================================
# 3. MODELO DE DESCUBRIMIENTO: ANOMALÍAS (Isolation Forest)
# ==========================================
print("🕵️ Ejecutando Detección de Anomalías (Isolation Forest)...")
# Buscamos alumnos que rompen la lógica Asistencia vs Notas
X_anom = df[['Asist_P2', 'Global_P2']].dropna()
iso_forest = IsolationForest(contamination=0.03, random_state=42)
iso_forest.fit(X_anom)

joblib.dump(iso_forest, 'model_anomalias.pkl')

# ==========================================
# 4. MODELOS DE IMPACTO CRUZADO (Árboles HD)
# ==========================================
print("🌳 Generando Árboles de Impacto Cruzado por Asignatura...")
troncales = {
    'Matematica': 'Matematica_Final',
    'Lenguaje': 'Lenguaje_Final',
    'Ciencias': 'Ciencias_Final',
    'Historia': 'Historia_Final'
}
apoyo = ['Ingles_P1', 'Tecnologia_P1', 'Artes_P1', 'Musica_P1', 'EdFisica_P1']

for nombre, col_troncal in troncales.items():
    if col_troncal in df.columns:
        df_tree = df.dropna(subset=[col_troncal]).copy()
        X_tree = df_tree[apoyo].fillna(4.0)
        y_tree = (df_tree[col_troncal] < 4.5).astype(int)
        
        # Árbol de profundidad 4 para balancear detalle y legibilidad
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
        tree.fit(X_tree, y_tree)
        
        # Visualización en Alta Definición
        plt.figure(figsize=(22, 12))
        plot_tree(tree, feature_names=apoyo, 
                  class_names=[f'Éxito {nombre}', f'Riesgo {nombre}'], 
                  filled=True, rounded=True, fontsize=11, precision=2)
        plt.title(f"Lógica de Impacto: {nombre}", fontsize=20)
        plt.savefig(f'arbol_impacto_{nombre}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar modelo para inferencia
        joblib.dump(tree, f'modelo_impacto_{nombre}.pkl')

# ==========================================
# 5. MODELO DE PUNTO DE QUIEBRE (Asistencia)
# ==========================================
print("📉 Analizando Punto de Quiebre de Asistencia...")
df_q = df.dropna(subset=['Global_P2', 'Asist_P2']).copy()
X_q = df_q[['Asist_P2']]
y_q = (df_q['Global_P2'] < 4.5).astype(int)

tree_q = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42)
tree_q.fit(X_q, y_q)

plt.figure(figsize=(12, 8))
plot_tree(tree_q, feature_names=['Asistencia'], class_names=['Estable', 'Riesgo'], 
          filled=True, rounded=True, fontsize=12, precision=1)
plt.title("Umbral Crítico de Asistencia", fontsize=18)
plt.savefig('arbol_quiebre.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*40)
print("✨ PROCESO COMPLETADO EXITOSAMENTE")
print("="*40)
print("Archivos generados:")
print("- model_final_regression.pkl (Cerebro Predictivo)")
print("- 4 Árboles de Impacto (.png)")
print("- 1 Árbol de Quiebre (.png)")
print("- model_anomalias.pkl (Detector de Outliers)")