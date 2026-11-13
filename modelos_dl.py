"""
ACA N.° 2 – Aprendizaje Automático (CUN)
Redes neuronales para clasificación de tumores de mama (UCI Breast Cancer Wisconsin – Diagnostic)

Contiene las dos arquitecturas comparadas en el proyecto:
  1) MLP profunda (perceptrón multicapa con normalización por lotes y dropout)
  2) FT-Transformer: Transformer para datos tabulares (Gorishniy et al., 2021): cada variable se convierte en un
     "token", se agrega un token [CLS] y varios bloques de autoatención (Vaswani et al., 2017) los combinan.
Lo usan el notebook (entrenamiento) y la app de Streamlit (inferencia), para que ambos coincidan.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")    # en la app se puede cambiar a "jax" (más liviano)
import json

import keras
import numpy as np
from keras import layers, ops

SEED = 42
BASES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
         "concave points", "symmetry", "fractal dimension"]
FEATURES = [f"mean {b}" for b in BASES] + [f"{b} error" for b in BASES] + [f"worst {b}" for b in BASES]


# ----------------------------------------------------------------------------- utilidades de datos
def estandarizar(X, media, desv):
    """Z-score con las estadísticas del conjunto de entrenamiento (evita fuga de información)."""
    return ((np.asarray(X, dtype="float32") - np.asarray(media, dtype="float32")) / np.asarray(desv, dtype="float32")).astype("float32")


def pesos_de_clase(y):
    """Pesos inversamente proporcionales a la frecuencia de cada clase (hay ~63 % benignos y ~37 % malignos)."""
    y = np.asarray(y).astype(int)
    n, k = len(y), 2
    return {c: float(n / (k * (y == c).sum())) for c in (0, 1)}


# ----------------------------------------------------------------------------- capas del Transformer tabular
@keras.saving.register_keras_serializable(package="aca2")
class FeatureTokenizer(layers.Layer):
    """Convierte cada variable numérica x_i en un vector (token) de d_model dimensiones: x_i * W_i + b_i."""

    def __init__(self, n_features, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_features, self.d_model = n_features, d_model

    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.n_features, self.d_model), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.n_features, self.d_model), initializer="zeros", name="b")

    def call(self, x):
        return ops.expand_dims(x, -1) * self.w + self.b

    def get_config(self):
        return {**super().get_config(), "n_features": self.n_features, "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="aca2")
class CLSToken(layers.Layer):
    """Antepone un token [CLS] entrenable; su salida final resume a todas las variables."""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls = self.add_weight(shape=(1, 1, self.d_model), initializer="glorot_uniform", name="cls")

    def call(self, x):
        return ops.concatenate([ops.zeros_like(x[:, :1, :]) + self.cls, x], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="aca2")
class TomarCLS(layers.Layer):
    def call(self, x):
        return x[:, 0, :]


def bloque_transformer(x, d_model, heads, ff, dropout):
    """Bloque Transformer (pre-normalización): autoatención multi-cabeza + red feed-forward, ambas con conexión residual."""
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)(h, h)
    x = layers.Add()([x, h])
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.Dense(ff, activation="gelu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(d_model)(h)
    return layers.Add()([x, h])


# ----------------------------------------------------------------------------- arquitecturas
def construir_mlp(n_features=30, unidades=(128, 64, 32), dropout=0.25, ruido=0.1):
    """MLP profunda. GaussianNoise = aumento de datos para datos tabulares (solo activo durante el entrenamiento)."""
    entrada = keras.Input((n_features,), name="variables")
    x = layers.GaussianNoise(ruido)(entrada)
    for u in unidades:
        x = layers.Dense(u, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    salida = layers.Dense(1, activation="sigmoid", name="prob_maligno")(x)
    return keras.Model(entrada, salida, name="MLP_profunda")


def construir_ft_transformer(n_features=30, d_model=32, heads=4, bloques=2, ff=64, dropout=0.1, ruido=0.1):
    """FT-Transformer: tokenizador de variables + [CLS] + bloques de autoatención + cabeza de clasificación."""
    entrada = keras.Input((n_features,), name="variables")
    x = layers.GaussianNoise(ruido)(entrada)
    x = FeatureTokenizer(n_features, d_model)(x)
    x = CLSToken(d_model)(x)
    for _ in range(bloques):
        x = bloque_transformer(x, d_model, heads, ff, dropout)
    x = TomarCLS()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(d_model, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(1, activation="sigmoid", name="prob_maligno")(x)
    return keras.Model(entrada, salida, name="FT_Transformer")


def compilar(modelo, lr=1e-3):
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                   metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return modelo


def entrenar(modelo, Xtr, ytr, Xva, yva, epocas=300, paciencia=30, batch=32):
    """Entrena con parada temprana (sobre la pérdida de validación) y reducción de la tasa de aprendizaje."""
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, paciencia // 3), min_lr=1e-5)]
    h = modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epocas, batch_size=batch, class_weight=pesos_de_clase(ytr),
                   callbacks=callbacks, verbose=0)
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def predecir(modelo, X, batch=256):
    """Probabilidad de malignidad. Se llama al modelo directamente (en lugar de model.predict) porque predict de Keras con el
    backend JAX no es seguro entre hilos, y Streamlit ejecuta cada interacción en un hilo distinto."""
    X = np.asarray(X, dtype="float32")
    salidas = [np.asarray(modelo(X[i:i + batch], training=False)).reshape(-1) for i in range(0, len(X), batch)]
    return np.concatenate(salidas)
