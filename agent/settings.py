"""Configurações do agente de monitoramento"""
import os

# Provedor do LLM, o prefixo do LiteLLM, como ollama, openai ou anthropic.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# Modelo usado pelos agentes que redigem os alertas e o plano em linguagem natural.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "text-qwen3-4b")

# URL do servidor LLM. Para Ollama local, o padrão abaixo.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Temperatura baixa porque as tarefas são de extração e narração fiel.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Timeout por chamada de LLM.
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# Limiar de decisão da CNN.
THRESHOLD = float(os.environ.get("FLOOD_THRESHOLD", "0.5"))

# Pasta raiz onde cada subpasta é uma "estação" de monitoramento.
STATIONS_DIR = os.environ.get(
    "STATIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "estacoes"),
)

# Recursos de resgate disponíveis.
# Ordenados do mais forte e escasso para o mais comum.
RECURSOS = [
    {"tipo": "Helicóptero", "qtd": 1},
    {"tipo": "Bote", "qtd": 2},
    {"tipo": "Equipe terrestre", "qtd": 3},
]
