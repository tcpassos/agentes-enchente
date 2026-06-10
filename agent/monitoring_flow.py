"""Núcleo dos agentes do centro de comando, feitos com CrewAI.

Na borda, o drone roda a CNN MobileNetV2 do vision.py, detecta enchente e emite
um evento. Não há LLM nessa parte. No centro de comando ficam os agentes de LLM
de Monitoramento, Triagem, Planejador e Navegador, acionados por evento.

Usado pela interface web app.py e pelo quadro de situação situation.py.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import yaml
from pydantic import BaseModel, Field
from crewai import Agent, LLM

from .settings import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_PROVIDER, LLM_TEMPERATURE, LLM_TIMEOUT,
)
from .vision import FloodResult

# Ranking de severidade, usado também pelo quadro de situação.
_SEV_RANK = {"alto": 3, "medio": 2, "baixo": 1, "normal": 0}

# Config dos agentes e das tarefas, separada do código.
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _load_yaml(name: str) -> dict:
    with open(os.path.join(_CONFIG_DIR, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


AGENTS_CFG = _load_yaml("agents.yaml")
TASKS_CFG = _load_yaml("tasks.yaml")


# --------------------------------------------------------------------------- #
# Modelos de dados
# --------------------------------------------------------------------------- #
class StationReport(BaseModel):
    station: str
    total_images: int = 0
    flooded_count: int = 0
    max_prob: float = 0.0
    mean_prob: float = 0.0
    severity: str = "normal"
    detections: list[dict] = Field(default_factory=list)


class RelatoVitima(BaseModel):
    """Pedido de socorro estruturado, extraído de uma mensagem pelo agente de Triagem."""

    local: str = "não informado"
    pessoas: int = 0
    necessidade: str = "não informado"
    urgencia: str = "media"
    resumo: str = ""


# --------------------------------------------------------------------------- #
# Severidade agregada por estação
# --------------------------------------------------------------------------- #
def _aggregate_severity(results: list[FloodResult]) -> str:
    if not results:
        return "normal"
    max_prob = max(r.probability for r in results)
    flooded = any(r.flooded for r in results)
    if max_prob >= 0.80:
        return "alto"
    if flooded:
        return "medio"
    if max_prob >= 0.30:
        return "baixo"
    return "normal"


def _build_report(station: str, results: list[FloodResult]) -> StationReport:
    n = len(results)
    return StationReport(
        station=station,
        total_images=n,
        flooded_count=sum(r.flooded for r in results),
        max_prob=max((r.probability for r in results), default=0.0),
        mean_prob=(sum(r.probability for r in results) / n) if n else 0.0,
        severity=_aggregate_severity(results),
        detections=[r.as_dict() for r in results],
    )


def build_recursos(**kwargs: int) -> list[dict]:
    """Monta a lista de recursos a partir de quantidades nomeadas.

    Por exemplo, build_recursos(Helicóptero=1, Bote=2, Equipe_terrestre=3). Tipos
    com quantidade 0 são omitidos. A interface usa isso para o operador definir o estoque.
    """
    return [
        {"tipo": tipo.replace("_", " "), "qtd": int(qtd)}
        for tipo, qtd in kwargs.items() if int(qtd) > 0
    ]


# --------------------------------------------------------------------------- #
# LLM (servidor e modelo), configurável em runtime pela interface
# --------------------------------------------------------------------------- #
def make_llm(
    model: str | None = None, base_url: str | None = None,
    provider: str | None = None, api_key: str | None = None,
) -> LLM:
    """Monta o LLM. Sem argumentos, usa os padrões de settings.py.

    O provider é o prefixo do LiteLLM, como ollama, openai ou anthropic, e o
    servidor não precisa ser Ollama. base_url e api_key são opcionais. Se ficarem
    vazios, usa o endpoint padrão do provedor e a chave do ambiente.
    """
    # base_url=None usa o default do config. base_url vazio usa o endpoint padrão do provedor.
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    kwargs: dict = {
        "model": f"{provider or LLM_PROVIDER}/{model or OLLAMA_MODEL}",
        "temperature": LLM_TEMPERATURE,   # baixa para o modelo não inventar
        "timeout": LLM_TIMEOUT,           # evita travar a interface se o servidor não responder
    }
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return LLM(**kwargs)


def listar_modelos(base_url: str | None = None) -> list[str]:
    """Lista os modelos pela API do Ollama. Retorna vazio se o servidor não
    responder ou não for Ollama. Nesse caso, digite o nome do modelo na mão."""
    import requests
    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        # O Ollama devolve nomes com ':latest'. Tiramos isso para casar com o default.
        return sorted(m["name"].removesuffix(":latest") for m in r.json().get("models", []))
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Agentes do centro de comando (LLM)
# --------------------------------------------------------------------------- #
def _build_agent(cfg_key: str, llm: LLM | str | None = None) -> Agent:
    cfg = AGENTS_CFG[cfg_key]
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm or make_llm(),
        max_iter=3,
        max_execution_time=LLM_TIMEOUT,
        verbose=False,
    )


def _build_alert_agent(llm: LLM | str | None = None) -> Agent:
    """Monitoramento: redige o resumo da situação."""
    return _build_agent("coordenador_emergencia", llm)


def _build_planner_agent(llm: LLM | str | None = None) -> Agent:
    """Planejador: justifica a alocação de recursos decidida pelo sistema."""
    return _build_agent("planejador_logistico", llm)


def _build_triage_agent(llm: LLM | str | None = None) -> Agent:
    """Triagem: extrai pedidos de socorro estruturados de mensagens cruas."""
    return _build_agent("triador_vitimas", llm)


def _build_navegador_agent(llm: LLM | str | None = None) -> Agent:
    """Navegador: redige a ordem tática para a unidade de uma missão."""
    return _build_agent("navegador_tatico", llm)


def _kickoff(agent: Agent, *args, **kwargs):
    """Executa agent.kickoff. Quando há um event loop ativo, como no notebook, ele
    devolve uma coroutine, então resolvemos para o resultado."""
    out = agent.kickoff(*args, **kwargs)
    if not asyncio.iscoroutine(out):
        return out
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(out)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, out).result()


# --------------------------------------------------------------------------- #
# Agente de Triagem: extrai pedidos de socorro de mensagens cruas
# --------------------------------------------------------------------------- #
_URGENCIAS = {"baixa", "media", "alta", "critica"}


def triar_mensagem(mensagem: str, agent: Agent | None = None) -> RelatoVitima | None:
    """Agente de Triagem: extrai um RelatoVitima estruturado de uma mensagem crua."""
    mensagem = (mensagem or "").strip()
    if not mensagem:
        return RelatoVitima(resumo="(mensagem vazia)")
    agent = agent or _build_triage_agent()
    prompt = TASKS_CFG["triar_mensagem"]["description"].format(mensagem=mensagem)
    try:
        relato = _kickoff(agent, prompt, response_format=RelatoVitima).pydantic
        if relato.urgencia not in _URGENCIAS:
            relato.urgencia = "media"
        return relato
    except Exception:
        return None
