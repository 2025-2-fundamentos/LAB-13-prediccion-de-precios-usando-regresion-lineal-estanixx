#
# Modelo de predicción de precios de vehículos usados
# Dataset con características de vehículos para pronóstico de precios
#

import os
import json
import gzip
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def leer_archivos_csv(ruta_entrenamiento: str, ruta_prueba: str):
    """Lee los datasets de entrenamiento y prueba desde archivos ZIP."""
    datos_entrenamiento = pd.read_csv(ruta_entrenamiento, compression="zip")
    datos_prueba = pd.read_csv(ruta_prueba, compression="zip")
    return datos_entrenamiento, datos_prueba


def transformar_datos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transforma el dataframe: crea columna Age y elimina columnas innecesarias."""
    df_copia = dataframe.copy()
    df_copia["Age"] = 2021 - df_copia["Year"]
    df_copia.drop(columns=["Year", "Car_Name"], inplace=True)
    return df_copia


def dividir_caracteristicas_objetivo(dataframe: pd.DataFrame):
    """Separa las características (X) de la variable objetivo (y)."""
    caracteristicas = dataframe.drop(columns="Present_Price")
    objetivo = dataframe["Present_Price"]
    return caracteristicas, objetivo


def crear_pipeline_ml(cols_categoricas, cols_numericas) -> Pipeline:
    """Construye el pipeline de machine learning con preprocesamiento y modelo."""
    transformador = ColumnTransformer([
        ("cat", OneHotEncoder(), cols_categoricas),
        ("num", MinMaxScaler(), cols_numericas)
    ])

    pipeline_completo = Pipeline(steps=[
        ("preprocesador", transformador),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regresor", LinearRegression())
    ])

    return pipeline_completo


def optimizar_hiperparametros(pipeline_base: Pipeline, caracteristicas, objetivo):
    """Optimiza hiperparámetros usando GridSearchCV con validación cruzada."""
    espacio_busqueda = {
        "selector__k": range(1, 15),
        "regresor__fit_intercept": [True, False],
        "regresor__positive": [True, False]
    }

    buscador = GridSearchCV(
        estimator=pipeline_base,
        param_grid=espacio_busqueda,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    return buscador.fit(caracteristicas, objetivo)


def persistir_modelo(modelo_entrenado, ubicacion: Path):
    """Serializa y comprime el modelo entrenado en formato gzip."""
    ubicacion.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ubicacion, "wb") as archivo:
        pickle.dump(modelo_entrenado, archivo)


def calcular_metricas(modelo_entrenado, X_entrenamiento, y_entrenamiento, 
                      X_prueba, y_prueba, archivo_destino: Path):
    """Calcula y guarda las métricas de desempeño del modelo."""
    predicciones_train = modelo_entrenado.predict(X_entrenamiento)
    predicciones_test = modelo_entrenado.predict(X_prueba)

    resultados_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_entrenamiento, predicciones_train)),
        "mse": float(mean_squared_error(y_entrenamiento, predicciones_train)),
        "mad": float(median_absolute_error(y_entrenamiento, predicciones_train)),
    }

    resultados_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_prueba, predicciones_test)),
        "mse": float(mean_squared_error(y_prueba, predicciones_test)),
        "mad": float(median_absolute_error(y_prueba, predicciones_test)),
    }

    archivo_destino.parent.mkdir(parents=True, exist_ok=True)
    with open(archivo_destino, "w", encoding="utf-8") as archivo:
        archivo.write(json.dumps(resultados_train) + "\n")
        archivo.write(json.dumps(resultados_test) + "\n")


def ejecutar_proceso():
    """Función principal que coordina todo el proceso de entrenamiento."""
    # Cargar datos
    datos_train, datos_test = leer_archivos_csv(
        "files/input/train_data.csv.zip",
        "files/input/test_data.csv.zip"
    )

    # Preprocesar
    datos_train = transformar_datos(datos_train)
    datos_test = transformar_datos(datos_test)

    # Separar X e y
    X_entrenamiento, y_entrenamiento = dividir_caracteristicas_objetivo(datos_train)
    X_prueba, y_prueba = dividir_caracteristicas_objetivo(datos_test)

    # Definir columnas categóricas y numéricas
    columnas_categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
    columnas_numericas = [col for col in X_entrenamiento.columns if col not in columnas_categoricas]
    
    # Crear pipeline
    pipeline_ml = crear_pipeline_ml(columnas_categoricas, columnas_numericas)

    # Optimizar modelo
    modelo_optimo = optimizar_hiperparametros(pipeline_ml, X_entrenamiento, y_entrenamiento)

    # Guardar modelo
    persistir_modelo(modelo_optimo, Path("files/models/model.pkl.gz"))

    # Evaluar y guardar métricas
    calcular_metricas(
        modelo_optimo,
        X_entrenamiento, y_entrenamiento,
        X_prueba, y_prueba,
        Path("files/output/metrics.json")
    )


if __name__ == "__main__":
    ejecutar_proceso()