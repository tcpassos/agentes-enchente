"""Quadro de situação (Common Operational Picture) do centro de comando.

Event store EM MEMÓRIA: as fontes (drones, triagem) PUBLICAM eventos que se
acumulam aqui; o centro de comando CONSOME o quadro para planejar e despachar.

Unifica as duas fontes num único conjunto de DEMANDAS priorizadas:
  - enchentes detectadas pelos drones  (prioridade = severidade)
  - pedidos de vítimas da triagem       (prioridade = urgência)

O Planejador aloca os recursos sobre essas demandas (de forma determinística,
respeitando o estoque) e o LLM narra o plano; o Navegador despacha as missões.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from .settings import RECURSOS
from .monitoring_flow import (
    RelatoVitima,
    make_llm, _build_planner_agent, _build_alert_agent, _build_navegador_agent,
    _pool_recursos, _SEV_RANK, TASKS_CFG,
)

# Ranking de urgência das vítimas (alinhado ao de severidade das enchentes).
_URG_RANK = {"critica": 4, "alta": 3, "media": 2, "baixa": 1}


class Demanda(BaseModel):
    """Necessidade de recurso de resgate (vinda de enchente OU de vítima)."""

    local: str
    tipo: str               # "enchente" | "vitima"
    prioridade: int         # 1..4 (maior = mais urgente)
    rotulo: str             # severidade (alto/medio) ou urgência (critica/alta/...)
    detalhe: str
    prob: float = 0.0
    pessoas: int = 0


class QuadroSituacao:
    """Estado acumulado do centro de comando (em memória)."""

    def __init__(self):
        self.deteccoes: list[dict] = []   # {local, prob, severidade, ts}
        self.vitimas: list[dict] = []     # {relato: RelatoVitima, ts}

    # --- Publicação de eventos (pelas fontes) ---
    def publicar_deteccao(self, local: str, prob: float, severidade: str):
        self.deteccoes.append({
            "local": local, "prob": prob, "severidade": severidade,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })

    def publicar_vitima(self, relato: RelatoVitima):
        self.vitimas.append({
            "relato": relato, "ts": datetime.now().isoformat(timespec="seconds"),
        })

    def limpar(self):
        self.deteccoes.clear()
        self.vitimas.clear()

    # --- Consumo (pelo centro de comando) ---
    def demandas(self) -> list[Demanda]:
        """Unifica enchentes + vítimas em demandas priorizadas (ordenadas)."""
        ds: list[Demanda] = []

        # Enchentes: agrega por local (mantém a maior probabilidade); só ocorrências.
        por_local: dict[str, dict] = {}
        for d in self.deteccoes:
            cur = por_local.get(d["local"])
            if cur is None or d["prob"] > cur["prob"]:
                por_local[d["local"]] = d
        for local, d in por_local.items():
            rank = _SEV_RANK.get(d["severidade"], 0)
            if rank < 2:        # só medio/alto contam como ocorrência
                continue
            ds.append(Demanda(
                local=local, tipo="enchente", prioridade=rank, rotulo=d["severidade"],
                prob=d["prob"], detalhe=f"enchente {d['severidade']} (prob {d['prob']:.0%})",
            ))

        # Vítimas: ignora urgência baixa (não-emergência, já filtrada na triagem).
        for v in self.vitimas:
            r: RelatoVitima = v["relato"]
            rank = _URG_RANK.get(r.urgencia, 2)
            if rank < 2:
                continue
            ds.append(Demanda(
                local=r.local, tipo="vitima", prioridade=rank, rotulo=r.urgencia,
                pessoas=r.pessoas, detalhe=f"{r.pessoas} pessoa(s) - {r.necessidade}",
            ))

        # Ordena por prioridade; no empate, VÍTIMA vem antes de enchente
        # (vidas em risco direto > área alagada), depois por nº de pessoas / prob.
        ds.sort(
            key=lambda x: (x.prioridade, 1 if x.tipo == "vitima" else 0, x.pessoas, x.prob),
            reverse=True,
        )
        return ds


# Instância única (singleton) usada pela interface.
QUADRO = QuadroSituacao()


# --------------------------------------------------------------------------- #
# Frota: unidades de resgate COM ESTADO (consumo + missões ativas)
# --------------------------------------------------------------------------- #
class MissaoAtiva(BaseModel):
    id: str
    recurso: str
    local: str
    tipo: str
    prioridade: int
    detalhe: str = ""
    ts: str
    briefing: str = ""          # ordem tática (gerada pelo Navegador, sob demanda)


class Frota:
    """Estado das unidades: total por tipo + missões ativas (unidades ocupadas)."""

    def __init__(self):
        self.total: dict[str, int] = {}
        self.ativas: list[MissaoAtiva] = []

    def definir_total(self, recursos: list[dict]):
        self.total = {r["tipo"]: int(r["qtd"]) for r in recursos}

    def ocupados(self) -> dict[str, int]:
        oc: dict[str, int] = {}
        for m in self.ativas:
            oc[m.recurso] = oc.get(m.recurso, 0) + 1
        return oc

    def disponiveis_lista(self) -> list[dict]:
        oc = self.ocupados()
        return [{"tipo": t, "qtd": max(0, tot - oc.get(t, 0))} for t, tot in self.total.items()]

    def locais_ativos(self) -> set[str]:
        return {m.local for m in self.ativas}

    def adicionar(self, recurso: str, demanda: Demanda):
        self.ativas.append(MissaoAtiva(
            id=uuid.uuid4().hex[:8], recurso=recurso, local=demanda.local,
            tipo=demanda.tipo, prioridade=demanda.prioridade, detalhe=demanda.detalhe,
            ts=datetime.now().isoformat(timespec="seconds"),
        ))

    def por_id(self, mid: str) -> MissaoAtiva | None:
        return next((m for m in self.ativas if m.id == mid), None)

    def concluir(self, mid: str):
        self.ativas = [m for m in self.ativas if m.id != mid]

    def concluir_todas(self) -> int:
        n = len(self.ativas)
        self.ativas.clear()
        return n

    def limpar(self):
        self.total.clear()
        self.ativas.clear()


FROTA = Frota()


def briefing_da_missao(m: MissaoAtiva, agent=None) -> str:
    """Agente Navegador: redige a ordem tática (briefing) para a unidade da missão."""
    prompt = TASKS_CFG["briefing_missao"]["description"].format(
        recurso=m.recurso, local=m.local, tipo=m.tipo,
        prioridade=m.prioridade, detalhe=m.detalhe,
    )
    agent = agent or _build_navegador_agent()
    try:
        return agent.kickoff(prompt).raw.strip()
    except Exception as e:
        print(f"  (LLM indisponível para o navegador: {e}; ordem por regras)")
        return (f"Ordem (regras): {m.recurso}, dirija-se a {m.local} "
                f"({m.tipo}, prioridade {m.prioridade}). {m.detalhe}.")


def frota_md(frota: Frota = FROTA) -> str:
    """Painel de monitoramento: unidades livres/ocupadas + missões ativas."""
    if not frota.total:
        return "_Frota ainda não definida. Clique em **Planejar e despachar**._"
    oc = frota.ocupados()
    linhas = [
        "### Recursos e equipes",
        "| Recurso | Total | Ocupadas | Livres |",
        "|---|:--:|:--:|:--:|",
    ]
    for t, tot in frota.total.items():
        o = oc.get(t, 0)
        linhas.append(f"| {t} | {tot} | {o} | {tot - o} |")
    if frota.ativas:
        linhas += ["", "**Missões ativas:**",
                   "| Recurso | Destino | Tipo | Despachada em |", "|---|---|---|---|"]
        for m in frota.ativas:
            linhas.append(f"| {m.recurso} | {m.local} | {m.tipo} | {m.ts} |")
    else:
        linhas += ["", "_Nenhuma missão ativa._"]
    return "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Alocação determinística sobre as demandas (respeita o estoque)
# --------------------------------------------------------------------------- #
def _alocar_demandas(demandas: list[Demanda], recursos: list[dict]):
    pool = _pool_recursos(recursos)
    aloc, i = [], 0
    for d in demandas:                       # já ordenadas por prioridade
        n = 2 if d.prioridade >= 3 else 1    # alta/crítica/alto -> 2; média/medio -> 1
        atribuidos = pool[i:i + n]
        i += len(atribuidos)
        aloc.append((d, atribuidos))
    return aloc


def despachar_alocacao(aloc, frota: Frota = FROTA):
    """Cria as missões ativas a partir de uma alocação JÁ CALCULADA (consome a frota)."""
    for d, recs in aloc:
        for recurso in recs:
            frota.adicionar(recurso, d)


def _texto_alocacao(aloc) -> str:
    """Descreve, em texto, a alocação determinística (recurso -> destino)."""
    linhas = []
    for d, recs in aloc:
        if recs:
            linhas.append(f"- {', '.join(recs)} -> {d.local} "
                          f"({d.tipo}, prioridade {d.prioridade}; {d.detalhe})")
        else:
            linhas.append(f"- {d.local} ({d.tipo}, prioridade {d.prioridade}): "
                          f"SEM RECURSO DISPONÍVEL")
    return "\n".join(linhas)


def _plano_regras_aloc(aloc) -> str:
    falta = any(not recs for _, recs in aloc)
    obs = ("Faltam recursos para atender todas as demandas." if falta
           else "Recursos suficientes para a demanda atual.")
    return "PLANO DE ALOCAÇÃO (regras):\n" + _texto_alocacao(aloc) + f"\nObs.: {obs}"


# --------------------------------------------------------------------------- #
# Planejador (LLM): JUSTIFICA a alocação determinística (não re-decide)
# --------------------------------------------------------------------------- #
def planejar_demandas(aloc, agent=None) -> str:
    if not aloc:
        return "Quadro sem ocorrências/vítimas ativas — nenhuma alocação necessária."
    prompt = TASKS_CFG["planejar_demandas"]["description"].format(
        alocacao=_texto_alocacao(aloc),
    )
    agent = agent or _build_planner_agent()
    try:
        return agent.kickoff(prompt).raw.strip()
    except Exception as e:
        print(f"  (LLM indisponível para o planejador: {e}; usando regras)")
        return _plano_regras_aloc(aloc)


# --------------------------------------------------------------------------- #
# Monitoramento: BRIEFING agregado da situação (1 chamada de LLM, sob demanda)
# --------------------------------------------------------------------------- #
def _briefing_regras(demandas: list[Demanda]) -> str:
    n_ench = sum(1 for d in demandas if d.tipo == "enchente")
    n_vit = sum(1 for d in demandas if d.tipo == "vitima")
    linhas = [f"BRIEFING (regras): {len(demandas)} demanda(s) ativa(s) — "
              f"{n_ench} enchente(s), {n_vit} vítima(s)."]
    for d in demandas[:3]:
        linhas.append(f"- Prioridade {d.prioridade}: [{d.tipo}] {d.local} ({d.detalhe}).")
    return "\n".join(linhas)


def gerar_briefing(demandas: list[Demanda], agent=None) -> str:
    """Agente de Monitoramento (coordenador_emergencia): redige UM briefing de
    situação agregado do quadro. Gatilho: sob demanda (não por evento)."""
    if not demandas:
        return "Quadro sem ocorrências/vítimas ativas — nada a reportar."
    situacao = "\n".join(
        f"  [{d.tipo}] {d.local} (prioridade {d.prioridade}; {d.detalhe})"
        for d in demandas
    )
    prompt = TASKS_CFG["briefing_situacao"]["description"].format(situacao=situacao)
    agent = agent or _build_alert_agent()
    try:
        return agent.kickoff(prompt).raw.strip()
    except Exception as e:
        print(f"  (LLM indisponível para o briefing: {e}; usando regras)")
        return _briefing_regras(demandas)


def resumo_md(quadro: QuadroSituacao = QUADRO) -> str:
    """Visão markdown do quadro de situação atual."""
    if not quadro.deteccoes and not quadro.vitimas:
        return ("_Quadro de situação vazio. Publique eventos nas abas **Sensor / Drone** "
                "e **Triagem**._")
    ds = quadro.demandas()
    linhas = [
        "### Quadro de situação",
        f"- Eventos de enchente (drones): **{len(quadro.deteccoes)}**",
        f"- Pedidos de vítima (triagem): **{len(quadro.vitimas)}**",
        f"- Demandas ativas (priorizadas): **{len(ds)}**",
        "",
        "| Prioridade | Tipo | Local | Detalhe |",
        "|:--:|---|---|---|",
    ]
    for d in ds:
        linhas.append(f"| {d.prioridade} | {d.tipo} | {d.local} | {d.detalhe} |")
    if not ds:
        linhas.append("| — | — | _nenhuma demanda ativa_ | — |")
    return "\n".join(linhas)
