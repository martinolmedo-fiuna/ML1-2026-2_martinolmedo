"""
=============================================================================
Trabajo Final ML1 - Predicción del Impacto de Anomalías Climáticas en Rendimiento
=============================================================================
Descripción:
  Este script implementa modelos de Machine Learning (Random Forest y XGBoost)
  para predecir el impacto de un escenario anómalo / estrés hídrico:
  ¿Cómo se afectaría el rendimiento (Yield kg/ha) si los días secos en la fase
  vegetativa ('Total_Dry_Days_Vegetative (days)') aumentan un 20% (u otro porcentaje)?

Metodología:
  1. Carga y preprocesamiento de 'df_procesado.xlsx'.
  2. Entrenamiento y validación cruzada (5-Fold CV) de modelos de regresión.
  3. Simulación contrafactual (What-if Scenario / Stress Testing):
     - Enfoque A: Ceteris Paribus directo sobre la variable.
     - Enfoque B: Propagación coherente sobre features derivadas (Dry_Stress_Vegetative = Max * Total).
  4. Análisis de sensibilidad continua (0% a +50% de días secos).
  5. Desagregación por Estado y detección de zonas con mayor vulnerabilidad.
  6. Generación de gráficos ejecutivos de impacto y guardado de resultados en Excel.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración estética de gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """Carga y valida el dataset de Excel."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")
    
    df = pd.read_excel(ruta_archivo)
    print(f"[INFO] Dataset cargado exitosamente. Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas.")
    return df


def preparar_features(df: pd.DataFrame):
    """
    Separa features (X) y target (y), descartando columnas de identificación.
    Detecta automáticamente nombres de columnas con posibles variaciones de caracteres especiales.
    """
    target_col = [c for c in df.columns if 'Yield' in c][0]
    
    # Descartar ID único (Location_Harvest) y Harvest (redundante con Year)
    cols_a_descartar = [target_col]
    for id_col in ['Location_Harvest', 'Harvest']:
        if id_col in df.columns:
            cols_a_descartar.append(id_col)
            
    X = df.drop(columns=cols_a_descartar)
    y = df[target_col]
    
    cat_cols = [c for c in ['State', 'Municipality'] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"[INFO] Target: '{target_col}' | Features numéricas: {len(num_cols)} | Categóricas: {cat_cols}")
    return X, y, target_col, num_cols, cat_cols


def crear_pipeline(modelo_tipo="rf"):
    """Construye un pipeline con preprocesamiento y el estimador seleccionado."""
    if modelo_tipo == "rf":
        estimador = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == "xgb":
        estimador = XGBRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("modelo_tipo debe ser 'rf' o 'xgb'")

    # Se definen los transformadores. Para modelos de árbol el escalamiento no es indispensable
    # pero mantiene la consistencia numérica y compatibilidad general.
    return estimador


def evaluar_modelos_cv(X, y, num_cols, cat_cols):
    """Evalúa modelos usando validación cruzada estratificada por k-fold."""
    print("\n" + "=" * 60)
    print(" 1. EVALUACIÓN Y VALIDACIÓN CRUZADA DE MODELOS (5-Fold CV)")
    print("=" * 60)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    modelos = {
        'Random Forest': Pipeline([('prep', preprocessor), ('reg', crear_pipeline("rf"))]),
        'XGBoost': Pipeline([('prep', preprocessor), ('reg', crear_pipeline("xgb"))])
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    for nombre, pipe in modelos.items():
        cv_res = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
        )
        r2 = cv_res['test_r2'].mean()
        mae = -cv_res['test_neg_mean_absolute_error'].mean()
        rmse = -cv_res['test_neg_root_mean_squared_error'].mean()
        resultados[nombre] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'pipe': pipe}
        print(f" -> {nombre:15s} | R²: {r2:.4f} | MAE: {mae:.2f} kg/ha | RMSE: {rmse:.2f} kg/ha")

    return resultados


def simular_anomalia_dias_secos(
    modelo_pipeline,
    X_original: pd.DataFrame,
    df_completo: pd.DataFrame,
    target_col: str,
    porcentaje_aumento: float = 0.20,
    propagar_interacciones: bool = True
) -> pd.DataFrame:
    """
    Ejecuta la simulación contrafactual del impacto del estrés hídrico.
    
    Parámetros:
      - modelo_pipeline: Modelo entrenado.
      - X_original: Matriz de variables predictoras.
      - df_completo: DataFrame con metadatos para cruzar (ej. State, Municipality).
      - porcentaje_aumento: Factor de anomalía (ej. 0.20 = +20%).
      - propagar_interacciones: Si es True, actualiza 'Dry_Stress_Vegetative'
                                calculada como (Max_Dry_Days_Vegetative * Total_Dry_Days_Vegetative).
    """
    col_dias_secos = 'Total_Dry_Days_Vegetative (days)'
    col_max_dias = 'Max_Dry_Days_Vegetative (days)'
    col_stress = [c for c in X_original.columns if 'Dry_Stress_Vegetative' in c]

    # Predicción base (sin alteración)
    pred_base = modelo_pipeline.predict(X_original)

    # Crear dataset con la perturbación (anomalía climática)
    X_anomalia = X_original.copy()
    X_anomalia[col_dias_secos] = X_anomalia[col_dias_secos] * (1.0 + porcentaje_aumento)

    # Coherencia física: si los días secos aumentan, el estrés compuesto también se recalcula
    if propagar_interacciones and col_stress and col_max_dias in X_original.columns:
        stress_name = col_stress[0]
        X_anomalia[stress_name] = X_anomalia[col_max_dias] * X_anomalia[col_dias_secos]

    # Predicción bajo escenario de estrés
    pred_anomalia = modelo_pipeline.predict(X_anomalia)

    # Cálculo de métricas de impacto
    delta_rendimiento = pred_anomalia - pred_base
    pct_impacto = (delta_rendimiento / pred_base) * 100

    df_resultado = pd.DataFrame({
        'State': df_completo['State'].values if 'State' in df_completo.columns else 'N/A',
        'Municipality': df_completo['Municipality'].values if 'Municipality' in df_completo.columns else 'N/A',
        'Year': df_completo['Year'].values if 'Year' in df_completo.columns else 0,
        'Rendimiento_Real': df_completo[target_col].values,
        'Rendimiento_Base_Pred': pred_base,
        'Rendimiento_Anomalia_Pred': pred_anomalia,
        'Delta_Rendimiento_kg_ha': delta_rendimiento,
        'Variacion_Porcentual_%': pct_impacto,
        'Dias_Secos_Original': X_original[col_dias_secos].values,
        'Dias_Secos_Simulado': X_anomalia[col_dias_secos].values
    })

    return df_resultado


def analisis_sensibilidad_curva(modelo_pipeline, X_original, porcentajes=None):
    """Evalúa la respuesta no lineal del rendimiento ante incrementos crecientes de sequía."""
    if porcentajes is None:
        porcentajes = np.linspace(0.0, 0.50, 11)  # Desde 0% hasta +50% en pasos de 5%

    col_dias_secos = 'Total_Dry_Days_Vegetative (days)'
    col_max_dias = 'Max_Dry_Days_Vegetative (days)'
    col_stress = [c for c in X_original.columns if 'Dry_Stress_Vegetative' in c]

    pred_base = modelo_pipeline.predict(X_original)
    rendimiento_promedio_base = pred_base.mean()

    curva = []
    for pct in porcentajes:
        X_mod = X_original.copy()
        X_mod[col_dias_secos] = X_mod[col_dias_secos] * (1.0 + pct)
        if col_stress and col_max_dias in X_original.columns:
            X_mod[col_stress[0]] = X_mod[col_max_dias] * X_mod[col_dias_secos]
        
        pred_mod = modelo_pipeline.predict(X_mod)
        delta_prom = (pred_mod - pred_base).mean()
        pct_prom = (delta_prom / rendimiento_promedio_base) * 100
        curva.append({
            'Incremento_%': pct * 100,
            'Delta_Promedio_kg_ha': delta_prom,
            'Variacion_Promedio_%': pct_prom,
            'Rendimiento_Esperado_kg_ha': pred_mod.mean()
        })

    return pd.DataFrame(curva)


def generar_reportes_y_graficos(df_sim, df_curva, salida_dir):
    """Genera gráficos ilustrativos y guarda resultados en formato Excel y PNG."""
    os.makedirs(salida_dir, exist_ok=True)

    # 1. Gráfico de Curva de Sensibilidad
    plt.figure(figsize=(9, 5))
    plt.plot(df_curva['Incremento_%'], df_curva['Variacion_Promedio_%'], marker='o', color='#d9534f', linewidth=2.5)
    plt.title("Sensibilidad del Rendimiento ante Días Secos en Fase Vegetativa", fontsize=13, fontweight='bold')
    plt.xlabel("Aumento de Días Secos en Fase Vegetativa (%)")
    plt.ylabel("Variación Promedio del Rendimiento (%)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Anotar el punto de +20%
    p20 = df_curva[df_curva['Incremento_%'] == 20.0]
    if not p20.empty:
        val = p20['Variacion_Promedio_%'].values[0]
        plt.annotate(
            f"+20% sequía -> {val:.2f}% ({p20['Delta_Promedio_kg_ha'].values[0]:.0f} kg/ha)",
            xy=(20, val),
            xytext=(22, val - 0.4),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )
    
    curva_path = os.path.join(salida_dir, "curva_sensibilidad_sequia.png")
    plt.savefig(curva_path, dpi=300)
    plt.close()

    # 2. Gráfico de Distribución del Impacto por Estado
    if 'State' in df_sim.columns and df_sim['State'].nunique() > 1:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_sim, x='State', y='Variacion_Porcentual_%', palette='Set2')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title("Distribución del Impacto de Anomalía (+20% Días Secos) por Estado", fontsize=13, fontweight='bold')
        plt.xlabel("Estado Productor")
        plt.ylabel("Caída en Rendimiento (%)")
        
        box_path = os.path.join(salida_dir, "impacto_por_estado.png")
        plt.savefig(box_path, dpi=300)
        plt.close()

    # 3. Guardar resultados detallados en Excel
    excel_path = os.path.join(salida_dir, "reporte_impacto_anomalias.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_sim.to_excel(writer, sheet_name='Impacto_Individual', index=False)
        df_curva.to_excel(writer, sheet_name='Curva_Sensibilidad', index=False)
        
        # Resumen por Estado
        if 'State' in df_sim.columns:
            resumen_estado = df_sim.groupby('State').agg(
                Casos=('Rendimiento_Real', 'count'),
                Rendimiento_Medio_Base=('Rendimiento_Base_Pred', 'mean'),
                Delta_Medio_kg_ha=('Delta_Rendimiento_kg_ha', 'mean'),
                Variacion_Media_Porc=('Variacion_Porcentual_%', 'mean'),
                Impacto_Max_Negativo_Porc=('Variacion_Porcentual_%', 'min')
            ).reset_index()
            resumen_estado.to_excel(writer, sheet_name='Resumen_Estado', index=False)

    print(f"\n[INFO] Gráficos y reporte guardados en:")
    print(f"  - {curva_path}")
    print(f"  - {excel_path}")


def main():
    # Rutas relativas y absolutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_excel = os.path.join(script_dir, "df_procesado.xlsx")
    salida_dir = os.path.join(script_dir, "resultados_simulacion")

    print("=" * 70)
    print(" INICIO DEL PROCESO DE SIMULACIÓN PREDICTIVA DE ANOMALÍAS")
    print("=" * 70)

    # 1. Carga de datos
    df = cargar_datos(archivo_excel)
    X, y, target_col, num_cols, cat_cols = preparar_features(df)

    # 2. Validación de Modelos
    cv_resultados = evaluar_modelos_cv(X, y, num_cols, cat_cols)

    # Seleccionar el mejor modelo según R² / MAE (generalmente Random Forest o XGBoost)
    mejor_nombre = max(cv_resultados, key=lambda k: cv_resultados[k]['R2'])
    print(f"\n[INFO] Modelo seleccionado para la simulación: '{mejor_nombre}'")

    pipeline_final = cv_resultados[mejor_nombre]['pipe']
    # Entrenar el pipeline con todos los datos disponibles para la simulación
    pipeline_final.fit(X, y)

    # 3. Simulación de Anomalía: +20% en Total_Dry_Days_Vegetative
    print("\n" + "=" * 60)
    print(" 2. SIMULACIÓN DE ANOMALÍA: +20% DÍAS SECOS EN FASE VEGETATIVA")
    print("=" * 60)
    df_sim = simular_anomalia_dias_secos(
        pipeline_final,
        X,
        df,
        target_col,
        porcentaje_aumento=0.20,
        propagar_interacciones=True
    )

    rend_base_medio = df_sim['Rendimiento_Base_Pred'].mean()
    delta_medio = df_sim['Delta_Rendimiento_kg_ha'].mean()
    pct_medio = df_sim['Variacion_Porcentual_%'].mean()
    peor_caso_pct = df_sim['Variacion_Porcentual_%'].min()
    peor_caso_delta = df_sim['Delta_Rendimiento_kg_ha'].min()

    print(f"Rendimiento Promedio Base:        {rend_base_medio:,.2f} kg/ha")
    print(f"Rendimiento Promedio Simulado:    {df_sim['Rendimiento_Anomalia_Pred'].mean():,.2f} kg/ha")
    print(f"Impacto Promedio (Delta):         {delta_medio:,.2f} kg/ha ({pct_medio:.2f}%)")
    print(f"Impacto Más Severo Registrado:    {peor_caso_delta:,.2f} kg/ha ({peor_caso_pct:.2f}%)")

    # 4. Curva de Sensibilidad Dinámica (0% a +50%)
    print("\n" + "=" * 60)
    print(" 3. ANÁLISIS DE SENSIBILIDAD MULTINIVEL (0% A +50% DE SEQUÍA)")
    print("=" * 60)
    df_curva = analisis_sensibilidad_curva(pipeline_final, X)
    for _, row in df_curva.iterrows():
        print(f" Sequía +{row['Incremento_%']:4.1f}% -> Variación: {row['Variacion_Promedio_%']:+6.2f}% ({row['Delta_Promedio_kg_ha']:+7.1f} kg/ha)")

    # 5. Generación de artefactos y gráficos
    generar_reportes_y_graficos(df_sim, df_curva, salida_dir)
    print("\n" + "=" * 70)
    print(" PROCESO COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    main()

