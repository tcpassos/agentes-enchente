"""Configurações do agente de monitoramento (Parte 2)."""
import os

# Provedor do LLM (prefixo do LiteLLM): "ollama", "openai", "anthropic", etc.
# Por padrão usamos Ollama (servidor local), mas o servidor não precisa ser Ollama.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# Modelo usado pelos agentes que redigem os alertas e o plano em linguagem natural.
# Para Ollama, é o nome do modelo em `ollama list`.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "text-qwen3-4b")

# URL do servidor LLM. Para Ollama local, o padrão abaixo.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Temperatura baixa: as tarefas são de extração/narração fiel; reduz a invenção.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Timeout (s) por chamada de LLM — evita travar a UI se o servidor não responder.
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# Limiar de decisão da CNN (probabilidade >= THRESHOLD => "enchente").
THRESHOLD = float(os.environ.get("FLOOD_THRESHOLD", "0.5"))

# Pasta raiz onde cada subpasta é uma "estação" de monitoramento.
STATIONS_DIR = os.environ.get(
    "STATIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "estacoes"),
)

# Recursos de resgate disponíveis (usados pelo agente PLANEJADOR para alocar).
# Ordenados do mais "forte"/escasso para o mais comum.
RECURSOS = [
    {"tipo": "Helicóptero", "qtd": 1},
    {"tipo": "Bote", "qtd": 2},
    {"tipo": "Equipe terrestre", "qtd": 3},
]
