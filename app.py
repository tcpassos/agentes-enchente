"""Interface web do sistema de monitoramento de enchentes, feita com Gradio.

Uso:
    python app.py

Abas:
  1. Sensor / Drone (borda): sobe uma imagem, a CNN detecta e publica o evento
     no quadro de situação.
  2. Triagem (mensagens): extrai um pedido de socorro estruturado de uma
     mensagem crua, pelo agente de Triagem.
  3. Centro de Comando: define a frota, planeja a alocação com o Planejador e
     gera o resumo da situação com o Monitoramento.
  4. Equipes ativas: gera a ordem tática de cada missão com o Navegador e conclui.
  5. Configurações: escolhe o provedor, o servidor LLM e o modelo dos agentes.

Antes de rodar, precisa do modelo flood_mobilenetv2.keras da Parte 1 e do Ollama ligado.
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime

# Saída em UTF-8 e telemetria desligada.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import gradio as gr

from agent.vision import classify_image
from agent.monitoring_flow import (
    _build_report, build_recursos,
    make_llm, listar_modelos, _build_alert_agent, _build_planner_agent,
    triar_mensagem, _build_triage_agent, _build_navegador_agent,
)
from agent.settings import (
    STATIONS_DIR, RECURSOS, OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_PROVIDER,
)
from agent.situation import (
    QUADRO, resumo_md, planejar_demandas, gerar_briefing,
    FROTA, _alocar_demandas, despachar_alocacao, frota_md, briefing_da_missao,
)

_DEFAULT_RES = {r["tipo"]: r["qtd"] for r in RECURSOS}


# --------------------------------------------------------------------------- #
# Configurações de LLM vindas da aba Configurações
# --------------------------------------------------------------------------- #
def _agentes_llm(server_url, model, provider, api_key):
    """Monta o agente de alerta e o de planejamento com o provedor, servidor e modelo escolhidos.

    server_url vazio usa o endpoint padrão do provedor. api_key vazio usa a chave do ambiente.
    """
    llm = make_llm(
        model=model or OLLAMA_MODEL,
        base_url=(server_url or "").strip(),   # vazio usa o endpoint padrão do provedor
        provider=provider or LLM_PROVIDER,
        api_key=(api_key or "").strip() or None,
    )
    return _build_alert_agent(llm), _build_planner_agent(llm)


def ajustar_url_por_provedor(provider):
    """Ao trocar de provedor, o Ollama usa a URL local e os outros usam o endpoint padrão."""
    return OLLAMA_BASE_URL if (provider or "").lower() == "ollama" else ""


def testar_conexao(server_url, model, provider):
    if (provider or "").lower() != "ollama":
        return (f"Provedor '{provider}': listagem automática só funciona com Ollama. "
                f"Digite o modelo manualmente. (Servidor: {server_url})")
    modelos = listar_modelos(server_url)
    if not modelos:
        return f"Falha ao acessar o servidor em {server_url}. Ele está rodando?"
    if model in modelos:
        return f"Conectado a {server_url}, {len(modelos)} modelo(s). Modelo '{model}' OK."
    return (f"Conectado a {server_url}, {len(modelos)} modelo(s). "
            f"Atenção: modelo '{model}' não está na lista.")


def atualizar_modelos(server_url, provider, modelo_atual):
    # A listagem automática só existe no Ollama. Para outros provedores,
    # limpa a lista e mantém o que estiver digitado.
    if (provider or "").lower() != "ollama":
        return gr.update(choices=[], value=modelo_atual)
    modelos = listar_modelos(server_url)
    if modelo_atual in modelos:
        valor = modelo_atual
    elif OLLAMA_MODEL in modelos:
        valor = OLLAMA_MODEL
    else:
        valor = modelos[0] if modelos else modelo_atual
    return gr.update(choices=modelos, value=valor)


# --------------------------------------------------------------------------- #
# Sensor / Drone (borda): só a CNN. Emite um evento, sem LLM.
# --------------------------------------------------------------------------- #
def detectar_imagem(imagem_path, local):
    if not imagem_path:
        return "Envie uma imagem do drone.", ""
    r = classify_image(imagem_path)
    rep = _build_report(local.strip() or "Estação", [r])
    classe = "ENCHENTE detectada" if r.flooded else "Sem enchente"
    resumo = (
        f"### {classe}\n"
        f"- **Probabilidade de enchente:** {r.probability:.1%}\n"
        f"- **Severidade:** {rep.severity.upper()}\n"
        f"- **Local:** {rep.station}"
    )
    # Evento que o drone publica no quadro de situação, só quando há ocorrência.
    if r.flooded:
        QUADRO.publicar_deteccao(rep.station, r.probability, rep.severity)
        evento = {
            "tipo": "deteccao_enchente",
            "local": rep.station,
            "prob_enchente": round(r.probability, 4),
            "severidade": rep.severity,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        evento_txt = ("// publicado no quadro de situação (aba Centro de Comando)\n"
                      + json.dumps(evento, ensure_ascii=False, indent=2))
    else:
        evento_txt = "// Sem enchente: a borda não publica evento de alerta."
    return resumo, evento_txt


# --------------------------------------------------------------------------- #
# Triagem e Centro de Comando (Monitoramento e Planejador)
# --------------------------------------------------------------------------- #
def triar(mensagem, provider, server_url, model, api_key):
    """Aba Triagem: o agente de LLM extrai um pedido de socorro estruturado da mensagem.

    É um gerador. Emite primeiro um aviso de processando, para dar retorno visual,
    e depois o resultado.
    """
    if not (mensagem or "").strip():
        yield "Digite uma mensagem para triar."
        return
    yield "_Triando a mensagem com o agente de Triagem..._"
    llm = make_llm(
        model=model or OLLAMA_MODEL, base_url=(server_url or "").strip(),
        provider=provider or LLM_PROVIDER, api_key=(api_key or "").strip() or None,
    )
    r = triar_mensagem(mensagem, _build_triage_agent(llm))
    if r is None:
        yield "Triagem indisponível: nenhum servidor LLM disponível."
        return
    QUADRO.publicar_vitima(r)   # publica no quadro de situação, que o centro consome
    nota = ("\n\n_Adicionado ao quadro de situação (aba Centro de Comando)._"
            if r.urgencia != "baixa"
            else "\n\n_Urgência baixa: registrado, mas não vira demanda de resgate._")
    yield (
        f"### Pedido de socorro (urgência: {r.urgencia.upper()})\n"
        f"- **Local:** {r.local}\n"
        f"- **Pessoas:** {r.pessoas}\n"
        f"- **Necessidade:** {r.necessidade}\n"
        f"- **Resumo:** {r.resumo}"
        f"{nota}"
    )


def limpar_quadro(eq_ver):
    QUADRO.limpar()
    FROTA.limpar()                 # reset completo, eventos e frota
    return resumo_md(QUADRO), frota_md(FROTA), eq_ver + 1


def planejar_centro(heli, bote, equipe, provider, server_url, model, api_key, eq_ver):
    """Centro de comando: lê o quadro e a frota disponível. Aloca apenas as unidades
    livres às demandas ainda não atendidas, e despachar já consome a frota."""
    FROTA.definir_total(build_recursos(Helicóptero=heli, Bote=bote, Equipe_terrestre=equipe))
    picture = resumo_md(QUADRO)
    # Demandas pendentes são as que ainda não têm unidade despachada, por local e tipo.
    ativas = FROTA.chaves_ativas()
    pendentes = [d for d in QUADRO.demandas() if (d.local, d.tipo) not in ativas]
    if not pendentes:
        return (picture, "Sem novas demandas a atender (quadro vazio ou todas já "
                "têm unidade despachada).", frota_md(FROTA), eq_ver + 1)

    disponiveis = FROTA.disponiveis_lista()
    # Uma decisão determinística, usada tanto no despacho quanto na narração do LLM,
    # garantindo que o plano descrito é o que foi realmente despachado.
    aloc = _alocar_demandas(pendentes, disponiveis)
    _, planner_agent = _agentes_llm(server_url, model, provider, api_key)
    plano = planejar_demandas(aloc, planner_agent)       # o LLM só justifica a alocação real
    despachar_alocacao(aloc, FROTA)                       # cria as equipes a partir da mesma alocação
    return picture, plano, frota_md(FROTA), eq_ver + 1


# --- Handlers por equipe (aba "Equipes ativas") ---
def _gerar_ordem_equipe(mid):
    """Fábrica: gera a ordem tática do Navegador para a missão mid, sob demanda."""
    def handler(provider, server_url, model, api_key):
        m = FROTA.por_id(mid)
        if m is None:
            yield "_Equipe não encontrada (talvez concluída)._"
            return
        yield "_Gerando ordem da missão (Navegador)..._"
        llm = make_llm(model=model or OLLAMA_MODEL, base_url=(server_url or "").strip(),
                       provider=provider or LLM_PROVIDER, api_key=(api_key or "").strip() or None)
        m.briefing = briefing_da_missao(m, _build_navegador_agent(llm))
        yield m.briefing
    return handler


def _concluir_equipe(mid):
    def handler(eq_ver):
        FROTA.concluir(mid)
        return eq_ver + 1          # incrementa para re-renderizar a lista de equipes
    return handler


def briefing_situacao(provider, server_url, model, api_key):
    """Agente de Monitoramento sob demanda: gera um resumo agregado do quadro."""
    ds = QUADRO.demandas()
    if not ds:
        yield "Quadro sem demandas, publique eventos primeiro."
        return
    yield "_Gerando o resumo da situação..._"
    alert_agent, _ = _agentes_llm(server_url, model, provider, api_key)
    yield gerar_briefing(ds, alert_agent)


def _exemplos():
    exemplos = []
    if os.path.isdir(STATIONS_DIR):
        for est in sorted(os.listdir(STATIONS_DIR)):
            pasta = os.path.join(STATIONS_DIR, est)
            if os.path.isdir(pasta):
                for img in sorted(os.listdir(pasta))[:2]:
                    exemplos.append([os.path.join(pasta, img), est])
    return exemplos


_MODELOS_INICIAIS = listar_modelos(OLLAMA_BASE_URL) or [OLLAMA_MODEL]

with gr.Blocks(title="Monitoramento de Enchentes") as demo:
    gr.Markdown(
        "# Sistema de Monitoramento de Enchentes\n"
        "**Borda (drone):** a CNN **MobileNetV2** detecta enchente na ponta e emite "
        "eventos. **Centro de comando:** os agentes **CrewAI + LLM** (Monitoramento "
        "e Planejador) são acionados por evento e geram alertas + plano de resgate."
    )

    # Componentes de configuração, definidos uma vez com render=False e posicionados
    # depois na aba Configurações, mas acessíveis pelos handlers de todas as abas.
    provider_dd = gr.Dropdown(
        choices=["ollama", "openai", "anthropic", "azure", "gemini"],
        value=LLM_PROVIDER, label="Provedor (LiteLLM)",
        allow_custom_value=True, render=False,
    )
    server_url = gr.Textbox(
        value=OLLAMA_BASE_URL, label="Servidor LLM (URL)",
        placeholder="vazio = endpoint padrão do provedor (ex.: OpenAI)", render=False,
    )
    model_dd = gr.Dropdown(
        choices=_MODELOS_INICIAIS,
        value=OLLAMA_MODEL if OLLAMA_MODEL in _MODELOS_INICIAIS else _MODELOS_INICIAIS[0],
        label="Modelo LLM", allow_custom_value=True, render=False,
    )
    api_key_tb = gr.Textbox(
        value="", label="API key (provedores pagos)", type="password",
        placeholder="vazio = usa a variável de ambiente (ex.: OPENAI_API_KEY)", render=False,
    )
    # Contador para forçar o re-render da lista de equipes ativas.
    eq_refresh = gr.State(0)

    with gr.Tab("Sensor / Drone (borda)"):
        gr.Markdown(
            "Simula **um frame** chegando do drone/sensor. A **borda roda só a CNN** "
            "(detecção, sem LLM): se houver enchente, ela **publica um evento** no "
            "barramento, que o centro de comando consome."
        )
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="Imagem do drone")
                local = gr.Textbox(value="Estação Central", label="Local / estação")
                btn1 = gr.Button("Detectar", variant="primary")
            with gr.Column():
                resumo = gr.Markdown()
                alerta1 = gr.Code(label="Evento publicado pela borda", language="json")
        btn1.click(detectar_imagem, [img, local], [resumo, alerta1])
        ex = _exemplos()
        if ex:
            gr.Examples(examples=ex, inputs=[img, local],
                        label="Exemplos das estações")

    with gr.Tab("Triagem (mensagens)"):
        gr.Markdown(
            "Outra **fonte de eventos**: mensagens de vítimas (WhatsApp/SMS/redes). "
            "Cole uma mensagem crua e o **agente de Triagem (LLM)** extrai e estrutura "
            "o pedido de socorro. Na arquitetura real, esse evento iria ao Planejador."
        )
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Mensagem recebida", lines=4,
                    placeholder="ex.: socorro, tem 4 pessoas presas no telhado da rua "
                                "das Flores 120, a água tá subindo rápido, tem uma criança",
                )
                btn_triar = gr.Button("Triar", variant="primary")
            with gr.Column():
                triagem_out = gr.Markdown()
        _msgs_exemplo = [
            "socorro tem 4 pessoas presas no telhado da rua das flores 120, a agua ta "
            "subindo rapido e tem uma criança pequena",
            "meu pai e cadeirante e a agua ja ta na altura do peito aqui na vila são jose, "
            "preciso de resgate urgente agora",
            "minha vó ta sozinha no bairro navegantes e a rua alagou toda, ela nao "
            "consegue sair de casa",
            "somos 8 pessoas no abrigo da escola municipal, sem agua potavel nem comida "
            "ha 2 dias",
            "alguem sabe se o mercado do centro abriu? queria comprar pão",
        ]
        gr.Examples(
            examples=[[m] for m in _msgs_exemplo],
            inputs=[msg],
            example_labels=_msgs_exemplo,
        )
        btn_triar.click(triar, [msg, provider_dd, server_url, model_dd, api_key_tb],
                        [triagem_out])

    with gr.Tab("Centro de Comando"):
        gr.Markdown(
            "### Centro de Comando\n"
            "Consome o **quadro de situação** acumulado pelas fontes (Drone e Triagem). "
            "Defina a frota e planeje: o **Planejador** decide a alocação, o **Navegador** "
            "despacha as equipes (aba *Equipes ativas*) e o **Monitoramento** redige o resumo."
        )
        timer_quadro = gr.Timer(2.0)   # refresh automático do quadro e da frota

        # Estado atual: quadro de situação + frota, lado a lado.
        with gr.Row(equal_height=True):
            with gr.Group():
                quadro_out = gr.Markdown(value=resumo_md())
            with gr.Group():
                frota_out = gr.Markdown(value=frota_md())

        # Controles: frota total + ações.
        with gr.Group():
            gr.Markdown("**Frota total e ações**")
            with gr.Row():
                heli = gr.Number(value=_DEFAULT_RES.get("Helicóptero", 1),
                                 label="Helicópteros", precision=0, minimum=0)
                bote = gr.Number(value=_DEFAULT_RES.get("Bote", 2),
                                 label="Botes", precision=0, minimum=0)
                equipe = gr.Number(value=_DEFAULT_RES.get("Equipe terrestre", 3),
                                   label="Equipes terrestres", precision=0, minimum=0)
            with gr.Row():
                btn2 = gr.Button("Planejar e despachar", variant="primary")
                btn_briefing = gr.Button("Gerar resumo da situação")
                btn_limpar = gr.Button("Limpar quadro", variant="secondary")

        # Saídas: Plano (Planejador) e Resumo (Monitoramento), separados.
        with gr.Row(equal_height=True):
            with gr.Group():
                gr.Markdown("#### Plano de alocação (Planejador)")
                plano_out = gr.Markdown(value="_Clique em **Planejar e despachar**._")
            with gr.Group():
                gr.Markdown("#### Resumo da situação (Monitoramento)")
                briefing_out = gr.Markdown(value="_Clique em **Gerar resumo da situação**._")

        timer_quadro.tick(lambda: (resumo_md(), frota_md()), None, [quadro_out, frota_out])
        btn_limpar.click(limpar_quadro, [eq_refresh], [quadro_out, frota_out, eq_refresh])
        btn2.click(
            planejar_centro,
            [heli, bote, equipe, provider_dd, server_url, model_dd, api_key_tb, eq_refresh],
            [quadro_out, plano_out, frota_out, eq_refresh],
        )
        btn_briefing.click(
            briefing_situacao,
            [provider_dd, server_url, model_dd, api_key_tb],
            [briefing_out],
        )

    with gr.Tab("Equipes ativas"):
        gr.Markdown(
            "Cada **equipe despachada** pelo Planejador aparece aqui. Gere a **ordem "
            "tática** (agente Navegador) e marque **concluída** ao terminar, isso "
            "libera a unidade de volta para a frota."
        )

        @gr.render(inputs=[eq_refresh], triggers=[demo.load, eq_refresh.change])
        def render_equipes(_v):
            ativas = list(FROTA.ativas)
            if not ativas:
                gr.Markdown("_Nenhuma equipe ativa. Despache no **Centro de Comando**._")
                return
            for m in ativas:
                with gr.Group():
                    gr.Markdown(
                        f"**{m.recurso} → {m.local}**  ·  {m.tipo} · prioridade "
                        f"{m.prioridade} · despachada {m.ts}"
                    )
                    gr.Markdown("Ordem da missão (Navegador):")
                    ordem = gr.Markdown(value=m.briefing or "_Ordem não gerada._")
                    with gr.Row():
                        bgen = gr.Button("Gerar ordem", size="sm")
                        bcon = gr.Button("Concluir missão", size="sm", variant="stop")
                    bgen.click(_gerar_ordem_equipe(m.id),
                               [provider_dd, server_url, model_dd, api_key_tb], [ordem])
                    bcon.click(_concluir_equipe(m.id), [eq_refresh], [eq_refresh])

    with gr.Tab("Configurações"):
        gr.Markdown(
            "Configure o **servidor LLM**, o **provedor** e o **modelo** usados pelos "
            "agentes. O padrão é Ollama local, mas o servidor não precisa ser Ollama "
            "(qualquer provedor suportado pelo LiteLLM). A CNN de detecção não depende disso.\n\n"
            "A lista automática de modelos só funciona com Ollama. Para outros "
            "provedores, digite o nome do modelo."
        )
        with gr.Row():
            provider_dd.render()
            server_url.render()
        with gr.Row():
            model_dd.render()
            btn_refresh = gr.Button("Atualizar lista de modelos")
        api_key_tb.render()
        btn_test = gr.Button("Testar conexão", variant="primary")
        status = gr.Markdown()
        # Ao trocar de provedor, ajusta a URL. Ollama usa local e os outros o padrão.
        provider_dd.change(ajustar_url_por_provedor, [provider_dd], [server_url])
        btn_refresh.click(atualizar_modelos, [server_url, provider_dd, model_dd], [model_dd])
        btn_test.click(testar_conexao, [server_url, model_dd, provider_dd], [status])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
