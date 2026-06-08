# Desafio IA — Etapa 2: Agente de Monitoramento de Enchentes

Detecção de enchentes em imagens com uma CNN (transfer learning) + agentes de
monitoramento que recebem os eventos e geram alertas, plano de alocação de recursos
e ordens de resgate.

- **Parte 1** — `model_train.ipynb`: treina e avalia um classificador de enchente com
  **MobileNetV2** (transfer learning) e compara com o **Xception** usado em aula.
- **Parte 2** — pasta `agent/` + `app.py`: agentes de monitoramento com **CrewAI** e um
  **servidor LLM** (LiteLLM — Ollama por padrão, mas qualquer provedor). A CNN faz a
  detecção; os agentes LLM transformam os eventos em um boletim operacional para a
  Defesa Civil.

Dataset: [Louisiana Flood 2016 (Kaggle)](https://www.kaggle.com/datasets/rahultp97/louisiana-flood-2016).

---

## 1. Instalação

```bash
pip install -r requirements.txt
```

> **Atenção (conflito de dependências):** instalar o `crewai` no mesmo ambiente do
> `tensorflow` costuma **rebaixar o `protobuf`** e quebrar o TF
> (`VersionError: ... gencode 6.31.1 runtime 5.29.6`). O `requirements.txt` fixa
> `protobuf==6.33.6`, com o qual **TF e CrewAI coexistem**. Se reinstalar o crewai,
> rode depois: `pip install "protobuf>=6.31,<7"`.

A Parte 2 precisa de um **servidor LLM**. O padrão é o [Ollama](https://ollama.com)
local com um modelo de **texto** (ex.: `ollama pull llama3.1`), mas qualquer provedor
suportado pelo LiteLLM serve (OpenAI, Anthropic, ...). Configure provedor, modelo,
servidor e API key em `agent/settings.py`, por env var, ou na aba **Configurações** do app.

---

## 2. Parte 1 — Treinar e avaliar a CNN

Abra e execute `model_train.ipynb` (ou rode tudo de uma vez):

```bash
jupyter nbconvert --to notebook --execute --inplace model_train.ipynb
```

O notebook baixa o dataset (kagglehub), monta o pipeline `tf.data` (224×224),
constrói o MobileNetV2 (base congelada + cabeça binária), treina, avalia (acurácia,
precision/recall/F1, matriz de confusão e latência) e **salva** o modelo em
`flood_mobilenetv2.keras`. O `.keras` é autossuficiente: augmentation e a
normalização do MobileNetV2 ([-1,1]) ficam embutidas como camadas — na inferência
basta passar a imagem redimensionada em [0,255].

Treina-se só a cabeça binária (extração de features, base congelada): para 270 imagens
de treino, fine-tunar os 2,2 M parâmetros da base arriscaria overfitting, então não há
fase de fine-tuning.

### Resultados (MobileNetV2 vs. Xception de aula)

Mesmo dataset (270 treino / 52 teste), execução local em CPU:

| Métrica                      | **MobileNetV2** (este trabalho) | Xception (notebook de aula) |
|------------------------------|:-------------------------------:|:---------------------------:|
| Acurácia (teste)             | **0,85** (~0,85–0,87)           | ~0,94                       |
| Parâmetros totais            | **2,26 M**                      | ~22 M                       |
| Tempo de treino              | **~120 s** / 60 épocas (~2 s/ép)| épocas de ~70–100 s         |
| Latência de inferência/imagem| **~40 ms** (~25 img/s)          | bem maior (modelo pesado)   |
| F1 classe "enchente"         | ~0,78                           | —                           |

> **Nota:** os números do Xception são os do notebook de aula (não re-treinado aqui).
> A comparação não é estritamente controlada: o Xception roda em 512×360, sem
> augmentation nem `class_weight`; este MobileNetV2 usa 224×224 com os dois.

**Conclusão da comparação:** o MobileNetV2 troca alguns pontos de acurácia por ser
**~10× mais leve** e bem mais rápido — perfil ideal para o agente de monitoramento
reativo (drones/edge), onde latência e custo computacional importam. O conjunto de
teste é pequeno (52 imagens), então ~1–2 pontos de acurácia são ruído estatístico.

---

## 3. Parte 2 — Agentes de monitoramento (CrewAI + servidor LLM)

```bash
python app.py
```

Os exemplos das estações (`estacoes/`) já vêm versionados; para regenerá-los a partir
do dataset, rode `python -m agent.setup_stations`.

A interface web (Gradio) tem cinco abas, organizadas pela arquitetura
**fontes de evento (borda) × centro de comando**:

- **Sensor / Drone (borda)** — arraste uma imagem: a CNN detecta enchente
  (probabilidade, classe, severidade) e, havendo ocorrência, publica um evento no
  quadro de situação. Sem LLM na borda.
- **Triagem (mensagens)** — cole uma mensagem crua de vítima (WhatsApp/SMS/rede): o
  **agente de Triagem (LLM)** extrai um pedido estruturado `{local, pessoas,
  necessidade, urgência}` e filtra o que não é emergência.
- **Centro de Comando** — agrega os eventos das fontes. O operador **define a frota**
  e planeja: o **Planejador** decide a alocação (determinística, respeitando o
  estoque) e o **Monitoramento** redige o briefing da situação.
- **Equipes ativas** — cada unidade despachada aparece aqui; o **Navegador** gera a
  ordem tática e a missão é concluída ao terminar (liberando a unidade de volta à frota).
- **Configurações** — escolhe o **provedor** (LiteLLM: ollama/openai/anthropic/...), o
  **servidor LLM (URL)**, o **modelo** e a **API key** (opcional). Para Ollama:
  `http://localhost:11434`; para provedores oficiais (OpenAI/Anthropic), deixe a URL
  vazia (endpoint padrão) e informe a API key — ou deixe-a vazia para usar a variável
  de ambiente (`OPENAI_API_KEY`...). A listagem automática de modelos só funciona com
  Ollama. A CNN de detecção não depende dessa configuração.

Sem servidor LLM, o Planejador, o Monitoramento e o Navegador caem para versões
**por regras** (determinísticas) e a demo não quebra; a **Triagem**, que depende de
extração de texto livre, fica **indisponível**.

### Arquitetura

```
   FONTES DE EVENTO (borda)               CENTRO DE COMANDO (event-driven)
┌────────────────────────────┐ eventos  ┌─────────────────────────────────────────┐
│ Drone: imagem → CNN         │ ───────► │ Monitoramento (LLM) → briefing/SITREP    │
│   MobileNetV2 (sem LLM)     │          │ Planejador (LLM)    → plano de alocação   │
│ Triagem: mensagem → LLM     │          │ Navegador (LLM)     → ordem por unidade   │
└────────────────────────────┘          └─────────────────────────────────────────┘
```

Na ponta (drone) só faz sentido a **CNN leve** (banda, latência, conectividade
intermitente); o **LLM** é caro/lento por frame e pertence ao **centro de comando**,
acionado **por evento** e de forma agregada. A alocação de recursos é
**determinística** (respeita o estoque e escolhe o recurso conforme o tipo de demanda)
e o LLM apenas **narra/justifica** a decisão, nunca a re-decide.

### Os agentes

| Agente | Tipo | Papel | Status |
|---|---|---|---|
| **Monitoramento** | Reativo | Detecta enchente (CNN) e redige o briefing da situação | ✅ Implementado |
| **Triagem** | Reativo | Extrai de uma mensagem crua um pedido estruturado `{local, pessoas, necessidade, urgência}` | ✅ Implementado |
| **Planejador** | Deliberativo | Justifica a alocação de recursos sobre as demandas priorizadas | ✅ Implementado |
| **Navegador** | Deliberativo | Redige a ordem tática por unidade; **sem engine de rotas** | 🟡 Mock |

### Testes do agente

`test_agente.ipynb` registra uma execução ponta-a-ponta dos quatro agentes (triagem,
detecção da CNN, plano, briefing e ordem tática). Defina provedor, modelo e servidor LLM
nas variáveis do topo do notebook e execute para capturar as saídas:

```bash
jupyter nbconvert --to notebook --execute --inplace test_agente.ipynb
```

> **Nota (limitação):** o quadro de situação e a frota são estado em memória
> compartilhado pelo processo — a demo é single-session (abas/usuários veem o mesmo quadro).
