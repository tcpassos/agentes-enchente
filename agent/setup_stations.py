"""Monta pastas de 'estações' de monitoramento a partir do dataset de teste.

Cada estação simula um local diferente recebendo imagens. Para uma demo
interessante, distribuímos um mix de cenas com e sem enchente:

  estacoes/
    Centro_Historico/      -> majoritariamente COM enchente (alerta alto)
    Bairro_Navegantes/     -> mix (alerta médio)
    Zona_Rural_Norte/      -> majoritariamente SEM enchente (situação normal)
"""
from __future__ import annotations

import os
import shutil

import kagglehub
import pandas as pd

from .settings import STATIONS_DIR

# Distribuição desejada por estação: (n_imagens_com_enchente, n_sem_enchente)
PLANO = {
    "Centro_Historico": (4, 1),
    "Bairro_Navegantes": (2, 2),
    "Zona_Rural_Norte": (0, 4),
}


def montar(seed: int = 42) -> str:
    data_dir = kagglehub.dataset_download("rahultp97/louisiana-flood-2016")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_img_dir = os.path.join(data_dir, "test")

    flooded = test_df[test_df["Flooded"] == 1].sample(frac=1, random_state=seed)
    normal = test_df[test_df["Flooded"] == 0].sample(frac=1, random_state=seed)
    fi = ni = 0

    if os.path.exists(STATIONS_DIR):
        shutil.rmtree(STATIONS_DIR)

    for estacao, (n_flood, n_normal) in PLANO.items():
        dest = os.path.join(STATIONS_DIR, estacao)
        os.makedirs(dest, exist_ok=True)
        escolhidas = (
            list(flooded["Image ID"].iloc[fi:fi + n_flood])
            + list(normal["Image ID"].iloc[ni:ni + n_normal])
        )
        fi += n_flood
        ni += n_normal
        for img in escolhidas:
            shutil.copy(os.path.join(test_img_dir, img), os.path.join(dest, img))
        print(f"  {estacao}: {len(escolhidas)} imagens "
              f"({n_flood} com enchente, {n_normal} sem)")

    print(f"\nEstações criadas em: {STATIONS_DIR}")
    return STATIONS_DIR


if __name__ == "__main__":
    montar()
