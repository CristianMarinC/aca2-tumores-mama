"""
ACA N.° 2 – Aprendizaje Automático (CUN)
Clasificación de tumores de mama con redes neuronales (MLP profunda y FT-Transformer) + base de datos SQL – App Streamlit
Dataset: UCI Breast Cancer Wisconsin (Diagnostic) – Wolberg, Street & Mangasarian (1995)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "jax")          # backend liviano para la app; los modelos se entrenaron con TensorFlow
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st

import modelos_dl as dl      # registra las capas personalizadas del Transformer

st.set_page_config(page_title="Redes neuronales · Cáncer de mama", page_icon="🧠", layout="wide")
BASE = Path(__file__).parent
MODELOS = ["MLP_profunda", "FT_Transformer"]
NOMBRES = {"MLP_profunda": "MLP profunda", "FT_Transformer": "FT-Transformer"}


def ruta(nombre):
    """Busca el archivo en la carpeta de la app o en 'artefactos_streamlit/' (la que genera el notebook)."""
    for carpeta in (BASE, BASE / "artefactos_streamlit"):
        if (carpeta / nombre).exists():
            return carpeta / nombre
    return None


faltan = [n for n in ("MLP_profunda.keras", "FT_Transformer.keras", "escalador.json", "metricas.json", "wdbc.db") if ruta(n) is None]
if faltan:
    st.error("Faltan archivos generados por el notebook: " + ", ".join(faltan) + ". Cópialos junto a `app.py` "
             "(están en `artefactos_streamlit.zip`).")
    st.stop()
DB = ruta("wdbc.db")


@st.cache_resource(show_spinner="Cargando las redes neuronales…")
def cargar_modelos():
    return {n: keras.saving.load_model(ruta(f"{n}.keras")) for n in MODELOS}


@st.cache_data
def cargar_json(nombre):
    with open(ruta(nombre), encoding="utf-8") as f:
        return json.load(f)


modelos, esc, met = cargar_modelos(), cargar_json("escalador.json"), cargar_json("metricas.json")
FEATURES = esc["features"]
COLS = [f.replace(" ", "_") for f in FEATURES]

BASE_ES = {"radius": "Radio", "texture": "Textura", "perimeter": "Perímetro", "area": "Área", "smoothness": "Suavidad",
           "compactness": "Compacidad", "concavity": "Concavidad", "concave points": "Puntos cóncavos",
           "symmetry": "Simetría", "fractal dimension": "Dimensión fractal"}
GRUPOS = {"mean": ("📏 Valores promedio (mean)", "mean {}"), "error": ("📐 Error estándar (error)", "{} error"),
          "worst": ("⚠️ Peores valores (worst)", "worst {}")}


# ------------------------------------------------------------------ acceso SQL
def conexion_lectura():
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=10)


def consulta(sql, params=()):
    with conexion_lectura() as con:
        return pd.read_sql(sql, con, params=params)


@st.cache_data(ttl=300)
def rangos():
    cols = ", ".join(f"MIN({c}) AS min_{c}, MAX({c}) AS max_{c}, AVG({c}) AS avg_{c}" for c in COLS)
    r = consulta(f"SELECT {cols} FROM pacientes").iloc[0]
    return {f: (float(r[f"min_{c}"]), float(r[f"max_{c}"]), float(r[f"avg_{c}"])) for f, c in zip(FEATURES, COLS)}


def ejemplo(diagnostico):
    d = consulta(f"SELECT {', '.join(COLS)} FROM pacientes WHERE particion = 'test' AND diagnostico = ? ORDER BY id LIMIT 1", (diagnostico,))
    return {f: float(d.iloc[0][c]) for f, c in zip(FEATURES, COLS)}


def guardar_predicciones(filas):
    try:
        con = sqlite3.connect(DB.as_posix(), timeout=10)
        con.executemany("INSERT INTO predicciones (fecha, modelo, umbral, probabilidad, clase, entrada) VALUES (?,?,?,?,?,?)", filas)
        con.commit(); con.close()
        return True
    except Exception as e:
        st.warning(f"No se pudo escribir en la base (en algunos servidores es de solo lectura): {e}")
        return False


R = rangos()
for f in FEATURES:
    st.session_state.setdefault(f"sl_{f}", R[f][2])


def cargar_valores(valores):
    for f in FEATURES:
        st.session_state[f"sl_{f}"] = float(valores[f])


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🧠 Configuración")
    st.caption(f"Dataset UCI WDBC · {sum(met['particion'].values())} pacientes · {len(FEATURES)} variables")
    st.caption(f"Modelo recomendado por validación: **{NOMBRES[met['recomendado']]}**")
    modo = st.radio("Umbral de decisión", ["Estándar (0,5)", "Ajustado con validación", "Personalizado"],
                    help="Probabilidad mínima de malignidad para marcar un caso como MALIGNO. Bajarlo detecta más malignos (más recall) "
                         "pero genera más falsas alarmas.")
    umbral_manual = st.slider("Umbral personalizado", 0.05, 0.95, 0.50, 0.01) if modo == "Personalizado" else None
    st.divider()
    st.warning("Herramienta académica. **No es un diagnóstico médico** y no reemplaza a un profesional de la salud.")


def umbral_de(n):
    if modo == "Estándar (0,5)":
        return 0.5
    if modo == "Ajustado con validación":
        return float(met["modelos"][n]["umbral_ajustado"])
    return umbral_manual


st.title("🧠 Clasificación de tumores de mama con redes neuronales")
st.markdown("Dos arquitecturas profundas, una **MLP** y un **Transformer para datos tabulares (FT-Transformer)**, entrenadas con el dataset "
            "UCI Breast Cancer Wisconsin y comparadas entre sí. Los datos se gestionan en una base **SQL (SQLite)**.")
tab1, tab2, tab3, tab4 = st.tabs(["🧍 Predicción", "🗄️ Datos (SQL)", "📈 Modelos y métricas", "ℹ️ Acerca del proyecto"])

# ================================================================== PREDICCIÓN
with tab1:
    st.subheader("Ingresa las medidas del paciente")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.button("📋 Cargar promedio", on_click=lambda: cargar_valores({f: R[f][2] for f in FEATURES}), width="stretch")
    c2.button("🟢 Ejemplo benigno", on_click=lambda: cargar_valores(ejemplo("B")), width="stretch")
    c3.button("🔴 Ejemplo maligno", on_click=lambda: cargar_valores(ejemplo("M")), width="stretch")
    c4.caption("Los ejemplos salen de la base SQL (conjunto de prueba).")
    for clave, (titulo, patron) in GRUPOS.items():
        with st.expander(titulo, expanded=(clave == "worst")):
            cols = st.columns(5)
            for i, b in enumerate(BASE_ES):
                f = patron.format(b); lo, hi, _ = R[f]
                with cols[i % 5]:
                    st.slider(BASE_ES[b], lo, hi, key=f"sl_{f}", step=(hi - lo) / 200 or 0.001, format="%.4f" if hi < 1 else "%.2f")

    x = np.array([[st.session_state[f"sl_{f}"] for f in FEATURES]], dtype="float32")
    z = dl.estandarizar(x, esc["media"], esc["desv"])
    probs = {n: float(dl.predecir(modelos[n], z)[0]) for n in MODELOS}
    clases = {n: probs[n] >= umbral_de(n) for n in MODELOS}

    st.divider()
    st.subheader("Resultado de las dos redes")
    cols = st.columns(2)
    for col, n in zip(cols, MODELOS):
        with col:
            st.markdown(f"#### {NOMBRES[n]}")
            (st.error if clases[n] else st.success)(f"### {'🔴 Posible MALIGNO' if clases[n] else '🟢 Posible BENIGNO'}")
            st.metric("Probabilidad de malignidad", f"{probs[n]:.1%}", help=f"Umbral aplicado: {umbral_de(n):.2f}")
            st.progress(min(max(probs[n], 0.0), 1.0))
            st.caption(f"Parámetros: {met['modelos'][n]['parametros']:,} · umbral {umbral_de(n):.2f}")
    if clases[MODELOS[0]] == clases[MODELOS[1]]:
        st.info("✅ Las dos redes coinciden en la clasificación.")
    else:
        st.warning("⚠️ Las dos redes **no coinciden**: es un caso límite que requiere revisión profesional.")
    if st.button("💾 Guardar esta predicción en la base SQL"):
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entrada = json.dumps({f: round(float(v), 5) for f, v in zip(FEATURES, x[0])})
        if guardar_predicciones([(ahora, n, umbral_de(n), probs[n], "M" if clases[n] else "B", entrada) for n in MODELOS]):
            st.success("Predicción registrada en la tabla `predicciones`.")

# ================================================================== DATOS SQL
with tab2:
    st.subheader("Base de datos SQLite del proyecto")
    tot = consulta("SELECT COUNT(*) AS total, SUM(diagnostico = 'M') AS malignos, SUM(diagnostico = 'B') AS benignos FROM pacientes").iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pacientes", int(tot.total)); m2.metric("Malignos", int(tot.malignos)); m3.metric("Benignos", int(tot.benignos))
    m4.metric("Tablas", len(consulta("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")))
    part = consulta("SELECT particion, diagnostico, COUNT(*) AS casos FROM pacientes GROUP BY particion, diagnostico")
    st.markdown("**Partición de los datos (registrada en SQL)**")
    st.bar_chart(part.pivot(index="particion", columns="diagnostico", values="casos"))

    st.markdown("#### Explorador de datos")
    f1, f2, f3 = st.columns(3)
    diag = f1.selectbox("Diagnóstico", ["Todos", "M (maligno)", "B (benigno)"])
    pa = f2.selectbox("Partición", ["Todas", "train", "val", "test"])
    n_filas = f3.slider("Filas a mostrar", 5, 100, 15)
    cols_ver = st.multiselect("Columnas", COLS, default=["mean_radius", "mean_texture", "mean_concavity", "worst_concave_points"])
    where, par = [], []
    if diag != "Todos":
        where.append("diagnostico = ?"); par.append(diag[0])
    if pa != "Todas":
        where.append("particion = ?"); par.append(pa)
    sql = f"SELECT id, diagnostico, particion{''.join(', ' + c for c in cols_ver)} FROM pacientes" + (" WHERE " + " AND ".join(where) if where else "") + f" ORDER BY id LIMIT {int(n_filas)}"
    st.code(sql, language="sql")
    datos = consulta(sql, tuple(par))
    st.dataframe(datos, width="stretch", hide_index=True)
    st.download_button("⬇️ Descargar esta vista (CSV)", datos.to_csv(index=False).encode("utf-8"), "vista_pacientes.csv", "text/csv")

    st.markdown("#### Consola SQL (solo lectura)")
    EJEMPLOS = {"Casos por diagnóstico": "SELECT diagnostico, COUNT(*) AS casos FROM pacientes GROUP BY diagnostico",
                "Promedios por diagnóstico": "SELECT diagnostico, ROUND(AVG(mean_radius), 2) AS radio, ROUND(AVG(mean_concavity), 3) AS concavidad, ROUND(AVG(worst_concave_points), 3) AS puntos_concavos FROM pacientes GROUP BY diagnostico",
                "Los 5 tumores de mayor radio": "SELECT id, diagnostico, mean_radius, mean_area FROM pacientes ORDER BY mean_radius DESC LIMIT 5",
                "Resultados de los experimentos": "SELECT modelo, particion, ROUND(accuracy, 4) AS accuracy, ROUND(recall, 4) AS recall, ROUND(f1, 4) AS f1, parametros FROM experimentos"}
    ej = st.selectbox("Consultas de ejemplo", list(EJEMPLOS))
    q = st.text_area("Consulta SQL", EJEMPLOS[ej], height=90, key=f"q_{ej}")
    if st.button("▶️ Ejecutar consulta"):
        ql = q.strip().rstrip(";").lower()
        if ";" in q.strip().rstrip(";") or not re.match(r"^(select|with)\b", ql) or re.search(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace|vacuum|detach)\b", ql):
            st.error("Solo se permiten consultas SELECT, de una sola sentencia.")
        else:
            try:
                with conexion_lectura() as con:
                    res = pd.read_sql(f"SELECT * FROM ({q.strip().rstrip(';')}) LIMIT 500", con)
                st.dataframe(res, width="stretch", hide_index=True); st.caption(f"{len(res)} fila(s) (máximo 500)")
            except Exception as e:
                st.error(f"Error en la consulta: {e}")
    st.markdown("#### Historial de predicciones guardadas")
    st.dataframe(consulta("SELECT id, fecha, modelo, umbral, ROUND(probabilidad, 4) AS probabilidad, clase FROM predicciones ORDER BY id DESC LIMIT 20"),
                 width="stretch", hide_index=True)

# ================================================================== MODELOS
with tab3:
    st.subheader("Arquitecturas")
    a1, a2 = st.columns(2)
    a1.markdown("**MLP profunda**  \n`Entrada(30) → GaussianNoise → [Dense → BatchNorm → ReLU → Dropout] ×3 (128, 64, 32) → Sigmoide`")
    a2.markdown("**FT-Transformer**  \n`Entrada(30) → GaussianNoise → Tokenizador de variables (30 tokens × 32) → [CLS] → 2 bloques Transformer (autoatención de 4 cabezas + feed-forward) → Sigmoide`")
    k = st.columns(4)
    for col, n in zip(k[:2], MODELOS):
        m = met["modelos"][n]
        col.metric(f"{NOMBRES[n]} · parámetros", f"{m['parametros']:,}")
        col.caption(f"{m['epocas']} épocas · {m['tiempo_s']} s de entrenamiento")
    k[2].metric("Partición", " / ".join(str(met["particion"][s_]) for s_ in ("train", "val", "test")))
    k[2].caption("entrenamiento / validación / prueba")
    k[3].metric("Modelo recomendado", NOMBRES[met["recomendado"]])
    k[3].caption("elegido con el conjunto de validación")

    st.markdown("#### Métricas (umbral 0,5)")
    filas = []
    for n in MODELOS:
        for sp in ("val", "test"):
            m = met["modelos"][n][sp]
            filas.append({"Modelo": NOMBRES[n], "Partición": sp, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"], "Pérdida": m["loss"], "FN": m["fn"], "FP": m["fp"]})
    st.dataframe(pd.DataFrame(filas).round(4), width="stretch", hide_index=True)
    st.markdown(f"#### Estabilidad con {met['n_semillas']} semillas (conjunto de prueba, media ± desviación)")
    est = []
    for n in MODELOS:
        s = met["modelos"][n]["semillas"]
        est.append({"Modelo": NOMBRES[n], **{k.capitalize(): f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in s.items()}})
    st.dataframe(pd.DataFrame(est), width="stretch", hide_index=True)
    st.caption("El conjunto de prueba es pequeño: diferencias de un caso cambian mucho las métricas, por eso se reporta la variación entre semillas.")

    st.markdown("#### Matrices de confusión (prueba)")
    cm_cols = st.columns(2)
    for col, n in zip(cm_cols, MODELOS):
        m = met["modelos"][n]["test"]
        col.markdown(f"**{NOMBRES[n]}**")
        col.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["Real benigno", "Real maligno"], columns=["Pred. benigno", "Pred. maligno"]), width="stretch")

    st.markdown("#### Curvas de aprendizaje")
    cc = st.columns(2)
    for col, n in zip(cc, MODELOS):
        h = met["modelos"][n]["historial"]
        col.markdown(f"**{NOMBRES[n]} – pérdida**")
        col.line_chart(pd.DataFrame({"entrenamiento": h["loss"], "validación": h["val_loss"]}))
        col.markdown(f"**{NOMBRES[n]} – exactitud**")
        col.line_chart(pd.DataFrame({"entrenamiento": h["accuracy"], "validación": h["val_accuracy"]}))
    st.markdown("#### Curvas ROC (prueba)")
    roc = pd.concat([pd.DataFrame({"fpr": met["modelos"][n]["roc"]["fpr"], "tpr": met["modelos"][n]["roc"]["tpr"], "modelo": NOMBRES[n]}) for n in MODELOS])
    st.line_chart(roc, x="fpr", y="tpr", color="modelo")
    st.markdown(f"#### Variables más influyentes ({NOMBRES[met['recomendado']]})")
    st.bar_chart(pd.Series(met["importancia_variables"]).sort_values())

# ================================================================== ACERCA
with tab4:
    st.markdown("""
**Asignatura:** Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN  
**Proyecto:** ACA N.° 2 – Modelado con redes neuronales y despliegue en Streamlit · **Pilar:** Salud

**Flujo metodológico:** (1) exploración y gestión de los datos en SQLite → (2) partición estratificada, estandarización, aumento de datos
(ruido gaussiano) y pesos de clase → (3) entrenamiento de una MLP profunda y un FT-Transformer con parada temprana →
(4) evaluación (loss, accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, estabilidad con semillas) → (5) esta aplicación.

**Datos:** 569 casos (357 benignos y 212 malignos) con 30 variables calculadas a partir de imágenes digitalizadas de aspirados con aguja fina.

**Referencias:** Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic ·
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS, 34*, 18932–18943.
""")
    st.warning("Prototipo académico. Sus resultados no constituyen un diagnóstico médico y deben ser valorados por un profesional de la salud. "
               "El conjunto de prueba es pequeño (86 pacientes) y no hay validación externa.")
