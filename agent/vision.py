"""Camada de visão do agente de monitoramento.

Carrega o modelo MobileNetV2 treinado na Parte 1 e oferece uma função simples
para classificar enchente a partir de um arquivo de imagem. Não depende do
CrewAI, então dá para testar sozinho.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import tensorflow as tf
import keras

from .settings import THRESHOLD

# Caminho do modelo salvo pelo notebook da Parte 1.
MODEL_PATH = os.environ.get(
    "FLOOD_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "flood_mobilenetv2.keras"),
)
IMAGE_SIZE = (224, 224)


@dataclass
class FloodResult:
    """Resultado da classificação de uma imagem."""

    image_path: str
    probability: float          # probabilidade de enchente, de 0 a 1
    flooded: bool               # decisão, probabilidade acima do threshold
    severity: str               # 'baixo', 'medio' ou 'alto'

    def as_dict(self) -> dict:
        return {
            "image": os.path.basename(self.image_path),
            "probability": round(self.probability, 4),
            "flooded": self.flooded,
            "severity": self.severity,
        }


@lru_cache(maxsize=1)
def _load_model() -> keras.Model:
    """Carrega o modelo uma única vez e guarda em cache."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o notebook model_train.ipynb (Parte 1) primeiro."
        )
    return keras.models.load_model(MODEL_PATH)


def _severity(prob: float) -> str:
    if prob >= 0.80:
        return "alto"
    if prob >= 0.50:
        return "medio"
    return "baixo"


def _load_image(path: str) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)          # escala [0, 255], a normalização fica dentro do modelo
    return tf.expand_dims(img, 0)


def classify_image(image_path: str, threshold: float = THRESHOLD) -> FloodResult:
    """Classifica uma imagem e retorna probabilidade, decisão e severidade."""
    model = _load_model()
    x = _load_image(image_path)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    return FloodResult(
        image_path=image_path,
        probability=prob,
        flooded=prob >= threshold,
        severity=_severity(prob),
    )


def classify_folder(folder: str, threshold: float = THRESHOLD) -> list[FloodResult]:
    """Classifica todas as imagens de uma pasta, simulando uma estação."""
    exts = (".png", ".jpg", ".jpeg")
    results = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            results.append(classify_image(os.path.join(folder, name), threshold))
    return results


if __name__ == "__main__":
    # Teste rápido: python -m agent.vision <pasta>
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    for r in classify_folder(target):
        print(r.as_dict())
