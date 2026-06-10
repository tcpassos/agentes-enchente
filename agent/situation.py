"""Quadro de situação do centro de comando.

Guarda os eventos em memória. As fontes, que são os drones e a triagem, publicam
eventos que se acumulam aqui. O centro de comando lê o quadro para planejar e despachar.

O Planejador distribui os recursos sobre as demandas de forma determinística,
respeitando o estoque, e o LLM narra o plano. O Navegador despacha as missões.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from .settings import RECURSOS
from .monitoring_flow import (
    RelatoVitima,
    make_llm, _build_planner_agent, _build_alert_agent, _build_navegador_agent,
    _kickoff, _SEV_RANK, TASKS_CFG,
)

# Ranking de urgência das vítimas, alinhado ao de severidade das enchentes.
_URG_RANK = {"critica": 4, "alta": 3, "media": 2, "baixa": 1}


class Demanda(BaseModel):
    local: str
    tipo: str               # "enchente" ou "vitima"
    prioridade: int         # 1 a 4, maior é mais urgente
    rotulo: str             # severidade (alto/medio) ou urgência (critica/alta/...)
    detalhe: str
    prob: float = 0.0
    pessoas: int = 0


class QuadroSituacao:
    """Estado acumulado do centro de comando"""

    def __init__(self):
        self.deteccoes: list[dict] = []   # {local, prob, severidade, ts}
        self.vitimas: list[dict] = []     # {relato: RelatoVitima, ts}

    # --- Publicação de eventos pelas fontes ---
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

    # --- Consumo pelo centro de comando ---
    def demandas(self) -> list[Demanda]:
        """Unifica enchentes e vítimas em demandas priorizadas e ordenadas."""
        ds: list[Demanda] = []

        # Enchentes: agrega por local mantendo a maior probabilidade. Só conta ocorrências.
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

        # Vítimas: ignora urgência baixa, que não é emergência e já foi filtrada na triagem.
        for v in self.vitimas:
            r: RelatoVitima = v["relato"]
            rank = _URG_RANK.get(r.urgencia, 2)
            if rank < 2:
                continue
            ds.append(Demanda(
                local=r.local, tipo="vitima", prioridade=rank, rotulo=r.urgencia,
                pessoas=r.pessoas, detalhe=f"{r.pessoas} pessoa(s) - {r.necessidade}",
            ))

        # Ordena por prioridade. No empate, a vítima vem antes da enchente,
        # porque vida em risco direto pesa mais que área alagada, e depois por
        # número de pessoas e probabilidade.
        ds.sort(
            key=lambda x: (x.prioridade, 1 if x.tipo == "vitima" else 0, x.pessoas, x.prob),
            reverse=True,
        )
        return ds


# Instância única usada pela interface.
QUADRO = QuadroSituacao()


# --------------------------------------------------------------------------- #
# Frota: unidades de resgate com estado, ou seja, estoque e missões ativas
# --------------------------------------------------------------------------- #
class MissaoAtiva(BaseModel):
    id: str
    recurso: str
    local: str
    tipo: str
    prioridade: int
    detalhe: str = ""
    ts: str
    briefing: str = ""          # ordem tática gerada pelo Navegador sob demanda


class Frota:
    """Estado das unidades: total por tipo e missões ativas."""

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

    def chaves_ativas(self) -> set[tuple[str, str]]:
        return {(m.local, m.tipo) for m in self.ativas}

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
    """Agente Navegador: redige a ordem tática para a unidade da missão."""
    prompt = TASKS_CFG["briefing_missao"]["description"].format(
        recurso=m.recurso, local=m.local, tipo=m.tipo,
        prioridade=m.prioridade, detalhe=m.detalhe,
    )
    agent = agent or _build_navegador_agent()
    ordem_regras = (f"Ordem (regras): {m.recurso}, dirija-se a {m.local} "
                    f"({m.tipo}, prioridade {m.prioridade}). {m.detalhe}.")
    try:
        ordem = _kickoff(agent, prompt).raw.strip()
        # Modelos pequenos às vezes devolvem só uma saudação, tipo "Vamos lá!".
        # Se a ordem vier curta demais para ser útil, usa a versão por regras.
        return ordem if len(ordem) >= 25 else ordem_regras
    except Exception as e:
        print(f"  (LLM indisponível para o navegador: {e}, ordem por regras)")
        return ordem_regras


def frota_md(frota: Frota = FROTA) -> str:
    """Painel de monitoramento com unidades livres e ocupadas e as missões ativas."""
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
# Alocação determinística sobre as demandas, respeitando o estoque
# --------------------------------------------------------------------------- #
# Preferência de recurso por tipo de demanda. Se não houver, usa qualquer disponível.
_PREF_RECURSO = {
    "vitima": ["Helicóptero", "Bote", "Equipe terrestre"],
    "enchente": ["Bote", "Equipe terrestre", "Helicóptero"],
}


def _alocar_demandas(demandas: list[Demanda], recursos: list[dict]):
    disp = {r["tipo"]: int(r["qtd"]) for r in recursos}
    aloc = []
    for d in demandas:                       # já ordenadas por prioridade
        n = 2 if d.prioridade >= 3 else 1    # alta, crítica ou alto recebem 2. média ou medio recebem 1
        pref = _PREF_RECURSO.get(d.tipo, [])
        ordem = pref + [t for t in disp if t not in pref]
        atribuidos = []
        for _ in range(n):
            escolha = next((t for t in ordem if disp.get(t, 0) > 0), None)
            if escolha is None:
                break
            disp[escolha] -= 1
            atribuidos.append(escolha)
        aloc.append((d, atribuidos))
    return aloc


def despachar_alocacao(aloc, frota: Frota = FROTA):
    """Cria as missões ativas a partir de uma alocação já calculada e consome a frota."""
    for d, recs in aloc:
        for recurso in recs:
            frota.adicionar(recurso, d)


def _texto_alocacao(aloc) -> str:
    """Descreve em texto a alocação determinística, no formato recurso para destino."""
    linhas = []
    for d, recs in aloc:
        if recs:
            linhas.append(f"- {', '.join(recs)} -> {d.local} "
                          f"({d.tipo}, prioridade {d.prioridade}, {d.detalhe})")
        else:
            linhas.append(f"- {d.local} ({d.tipo}, prioridade {d.prioridade}): "
                          f"SEM RECURSO DISPONÍVEL")
    return "\n".join(linhas)


def _plano_regras_aloc(aloc) -> str:
    falta = any(not recs for _, recs in aloc)
    obs = ("Faltam recursos para atender todas as demandas." if falta
           else "Recursos suficientes para a demanda atual.")
    return "Plano de alocação (regras):\n" + _texto_alocacao(aloc) + f"\nObs.: {obs}"


# --------------------------------------------------------------------------- #
# Planejador (LLM): justifica a alocação determinística, sem re-decidir
# --------------------------------------------------------------------------- #
def planejar_demandas(aloc, agent=None) -> str:
    if not aloc:
        return "Quadro sem ocorrências ou vítimas ativas, nenhuma alocação necessária."
    prompt = TASKS_CFG["planejar_demandas"]["description"].format(
        alocacao=_texto_alocacao(aloc),
    )
    agent = agent or _build_planner_agent()
    try:
        return _kickoff(agent, prompt).raw.strip()
    except Exception as e:
        print(f"  (LLM indisponível para o planejador: {e}, usando regras)")
        return _plano_regras_aloc(aloc)


# --------------------------------------------------------------------------- #
# Monitoramento: resumo agregado da situação, uma chamada de LLM sob demanda
# --------------------------------------------------------------------------- #
def _briefing_regras(demandas: list[Demanda]) -> str:
    n_ench = sum(1 for d in demandas if d.tipo == "enchente")
    n_vit = sum(1 for d in demandas if d.tipo == "vitima")
    linhas = [f"Resumo (regras): {len(demandas)} demanda(s) ativa(s). "
              f"{n_ench} enchente(s), {n_vit} vítima(s)."]
    for d in demandas[:3]:
        linhas.append(f"- Prioridade {d.prioridade}: [{d.tipo}] {d.local} ({d.detalhe}).")
    return "\n".join(linhas)


def gerar_briefing(demandas: list[Demanda], agent=None) -> str:
    """Agente de Monitoramento: redige um resumo da situação agregado do quadro, sob demanda."""
    if not demandas:
        return "Quadro sem ocorrências ou vítimas ativas, nada a reportar."
    situacao = "\n".join(
        f"  [{d.tipo}] {d.local} (prioridade {d.prioridade}, {d.detalhe})"
        for d in demandas
    )
    prompt = TASKS_CFG["briefing_situacao"]["description"].format(situacao=situacao)
    agent = agent or _build_alert_agent()
    try:
        return _kickoff(agent, prompt).raw.strip()
    except Exception as e:
        print(f"  (LLM indisponível para o monitoramento: {e}, usando regras)")
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
        linhas.append("| - | - | _nenhuma demanda ativa_ | - |")
    return "\n".join(linhas)
