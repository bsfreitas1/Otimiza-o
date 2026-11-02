#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras do trabalho sobre alocacao de gas lift.

Sao geradas tres figuras em PNG a 300 dpi, dimensionadas para a largura de
coluna do resumo estendido. O titulo de cada uma fica na legenda do LaTeX, e
nao dentro da imagem.

    fig1_curvas.png        as duas curvas que descrevem cada poco
    fig2_alocacao.png      alocacao otima e preco-sombra do gas
    fig3_escala.png        custo dos metodos e quebra da enumeracao

Uso:
    python gaslift.py --figuras
    python figuras.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -- paleta categorica, validada para visao normal e para daltonismo ---------
AZUL = "#2563EB"      # pocos de tipo a ou b
LARANJA = "#D9690F"   # pocos de tipo c ou d
VERDE = "#0E8F80"     # terceira serie, usada so na figura 3
TINTA = "#1F2937"     # texto
CINZA = "#9AA3AF"     # grade e eixos


def _estilo() -> None:
    """Estilo sobrio, compativel com o corpo de texto em Times do template."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.edgecolor": CINZA,
        "axes.linewidth": 0.7,
        "axes.labelcolor": TINTA,
        "text.color": TINTA,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": CINZA,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _limpa(ax) -> None:
    """Remove as bordas superior e direita e deixa a grade recessiva."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)


# ===========================================================================
#  FIGURA 1  As duas curvas de cada poco
# ===========================================================================

def fig_curvas(pocos: dict, caminho: str = "fig1_curvas.png") -> str:
    """
    (a) curva de desempenho, oleo contra gas injetado
    (b) curva economica, lucro diario contra gas injetado

    A cor separa os pocos pelo tipo: em azul os que sao rentaveis desde a
    primeira unidade de gas, em laranja os que so se pagam acima de uma
    injecao minima, que sao a origem da parte nao convexa do problema.
    """
    _estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75))

    qg = np.linspace(0.0, 3.2, 400)
    for nome, p in pocos.items():
        tipo_cd = p.qg_min_economico is not None
        cor = LARANJA if tipo_cd else AZUL
        traco = "-" if tipo_cd else (0, (4, 1.6))
        lw = 1.6 if tipo_cd else 1.0

        ax1.plot(qg, [p.oleo(float(x)) for x in qg], color=cor,
                 linestyle=traco, linewidth=lw, zorder=3)
        ax1.plot(p.qg_max_oleo, p.oleo(p.qg_max_oleo), "o", color=cor,
                 markersize=3.4, markeredgecolor="white",
                 markeredgewidth=0.6, zorder=4)

        ax2.plot(qg, [p.lucro(float(x)) for x in qg], color=cor,
                 linestyle=traco, linewidth=lw, zorder=3)
        ax2.plot(p.qg_max_lucro, p.lucro_max, "o", color=cor, markersize=3.4,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        if tipo_cd:
            qmin = p.qg_min_economico
            ax2.plot(qmin, 0.0, "v", color=cor, markersize=5.2,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6,
                     clip_on=False)

    # rotulos diretos nos dois pocos de tipo c ou d
    for nome in ("W7", "W8"):
        p = pocos[nome]
        ax1.annotate(nome, (p.qg_max_oleo, p.oleo(p.qg_max_oleo)),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7.5, color=LARANJA, fontweight="bold")
        ax2.annotate(nome, (p.qg_max_lucro, p.lucro_max),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7.5, color=LARANJA, fontweight="bold")

    ax1.set_xlabel(r"gás injetado $q_g$  (10$^4$ m$^3$/d)")
    ax1.set_ylabel(r"óleo produzido $q_o$  (m$^3$/d)")
    ax1.set_xlim(0, 3.2)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(MultipleLocator(1.0))

    ax2.axhline(0.0, color=TINTA, linewidth=0.7, zorder=2)
    ax2.set_xlabel(r"gás injetado $q_g$  (10$^4$ m$^3$/d)")
    ax2.set_ylabel(r"lucro diário $y$  (US\$/d)")
    ax2.set_xlim(0, 3.2)
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))

    for ax, letra in ((ax1, "(a)"), (ax2, "(b)")):
        _limpa(ax)
        ax.text(0.015, 1.04, letra, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=TINTA)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=AZUL, linestyle=(0, (4, 1.6)), linewidth=1.0,
               label="tipos a e b, rentáveis desde o início"),
        Line2D([], [], color=LARANJA, linestyle="-", linewidth=1.6,
               label="tipos c e d, exigem injeção mínima"),
        Line2D([], [], color=LARANJA, marker="v", linestyle="none",
               markersize=4.2, label=r"limiar de rentabilidade $q_{g,\min}$"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3,
        frameon=False, handlelength=2.2, columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# ===========================================================================
#  FIGURA 2  Alocacao otima e preco-sombra
# ===========================================================================

def fig_alocacao(pocos: dict, caminho: str = "fig2_alocacao.png") -> str:
    """
    (a) quanto cada poco recebe conforme o gas disponivel cresce
    (b) multiplicador de Lagrange, que e o preco-sombra do gas

    O painel (a) mostra os pocos entrando um a um na alocacao, e o (b) mostra
    que isso acontece exatamente quando o preco-sombra cruza a produtividade
    marginal de cada poco em vazao nula.
    """
    from gaslift import ESQUEMAS, resolve_kkt

    _estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75))

    ativos = ESQUEMAS[1]
    grade = np.linspace(0.25, 16.6, 320)
    Q = np.zeros((len(grade), len(ativos)))
    lam = np.zeros(len(grade))
    for k, Qmax in enumerate(grade):
        q, L = resolve_kkt(pocos, ativos, float(Qmax), economico=True)
        Q[k], lam[k] = q, L

    for j, nome in enumerate(ativos):
        p = pocos[nome]
        tipo_cd = p.qg_min_economico is not None
        cor = LARANJA if tipo_cd else AZUL
        traco = "-" if tipo_cd else (0, (4, 1.6))
        ax1.plot(grade, Q[:, j], color=cor, linestyle=traco,
                 linewidth=1.6 if tipo_cd else 1.0, zorder=3)

    # rotula apenas os dois pocos que governam a parte nao convexa; os demais
    # ficam identificados pela cor, cujo significado a legenda explica
    for nome in ("W7", "W8"):
        j = ativos.index(nome)
        ax1.annotate(nome, (grade[-1], Q[-1, j]),
                     textcoords="offset points", xytext=(5, -2.5),
                     fontsize=7.5, color=LARANJA, fontweight="bold")

    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([], [], color=AZUL, linestyle=(0, (4, 1.6)), linewidth=1.0,
               label="tipos a e b"),
        Line2D([], [], color=LARANJA, linestyle="-", linewidth=1.6,
               label="tipos c e d"),
    ], frameon=False, loc="upper left", handlelength=2.0, fontsize=7.5)

    ax1.set_xlabel(r"gás disponível $Q_{\max}$  (10$^4$ m$^3$/d)")
    ax1.set_ylabel(r"gás alocado ao poço $q_{g,i}$  (10$^4$ m$^3$/d)")
    ax1.set_xlim(0, 18.4)
    ax1.set_ylim(bottom=0)

    # painel (b): preco-sombra e os pontos de entrada de cada poco
    ax2.plot(grade, lam, color=VERDE, linewidth=1.8, zorder=3)
    entradas = []
    for j, nome in enumerate(ativos):
        entrou = np.flatnonzero(Q[:, j] > 1e-9)
        if entrou.size and entrou[0] > 0:
            entradas.append((entrou[0], nome))
    entradas.sort()
    # escada de rotulos a direita da curva, com linha fina ligando cada um ao
    # seu ponto de entrada, de modo que nenhum rotulo se sobreponha a outro
    topo = float(lam.max())
    for i, (k, nome) in enumerate(entradas):
        ax2.plot(grade[k], lam[k], "o", color=TINTA, markersize=3.4,
                 markeredgecolor="white", markeredgewidth=0.7, zorder=6)
        x_rot = 6.6 + 1.55 * i
        y_rot = topo * (0.96 - 0.105 * i)
        ax2.annotate(nome, xy=(grade[k], lam[k]), xytext=(x_rot, y_rot),
                     fontsize=7.2, color=TINTA, va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color=CINZA,
                                     linewidth=0.6,
                                     shrinkA=1.5, shrinkB=3.5))

    ax2.set_xlabel(r"gás disponível $Q_{\max}$  (10$^4$ m$^3$/d)")
    ax2.set_ylabel(r"preço-sombra $\lambda$  (US\$/10$^4$ m$^3$)")
    ax2.set_xlim(0, 18.4)
    ax2.set_ylim(bottom=0)

    for ax, letra in ((ax1, "(a)"), (ax2, "(b)")):
        _limpa(ax)
        ax.text(0.015, 1.04, letra, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=TINTA)

    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# ===========================================================================
#  FIGURA 3  Escalabilidade
# ===========================================================================

def fig_escala(caminho: str = "fig3_escala.png") -> str:
    """
    (a) tempo de uma resolucao do problema continuo, pelos tres metodos
    (b) tempo total da enumeracao dos pocos de tipo c e d

    O painel (b) usa escala logaritmica e uma linha de referencia em uma hora,
    para situar onde o procedimento do artigo deixa de ser praticavel.
    """
    import time
    from gaslift import (instancia_sintetica, resolve_kkt, resolve_penalidade,
                         resolve_sqp)

    _estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75))
    rng = np.random.default_rng(20260120)

    # ---- painel (a) -------------------------------------------------------
    tamanhos = [8, 25, 50, 100, 250, 1_000, 5_000, 20_000, 100_000]
    t_kkt, t_pen, t_sqp, n_sqp = [], [], [], []
    for n in tamanhos:
        p = instancia_sintetica(n, 0.0, rng)
        at = list(p)
        Qmax = 0.35 * sum(p[w].qg_max_lucro for w in at)

        t0 = time.perf_counter()
        resolve_kkt(p, at, Qmax, economico=True)
        t_kkt.append((time.perf_counter() - t0) * 1e3)

        if n <= 20_000:
            t0 = time.perf_counter()
            resolve_penalidade(p, at, Qmax, economico=True)
            t_pen.append((time.perf_counter() - t0) * 1e3)
        if n <= 250:
            t0 = time.perf_counter()
            resolve_sqp(p, at, Qmax, economico=True)
            t_sqp.append((time.perf_counter() - t0) * 1e3)
            n_sqp.append(n)

    ax1.plot(tamanhos, t_kkt, "o-", color=AZUL, linewidth=1.6, markersize=3.6,
             markeredgecolor="white", markeredgewidth=0.5,
             label="KKT por bisseção", zorder=4)
    ax1.plot(tamanhos[:len(t_pen)], t_pen, "s--", color=LARANJA, linewidth=1.6,
             markersize=3.6, markeredgecolor="white", markeredgewidth=0.5,
             label="penalidade mista", zorder=4)
    ax1.plot(n_sqp, t_sqp, "^:", color=VERDE, linewidth=1.6, markersize=4.2,
             markeredgecolor="white", markeredgewidth=0.5,
             label="SQP", zorder=4)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("número de poços")
    ax1.set_ylabel("tempo de uma resolução (ms)")
    ax1.legend(frameon=False, loc="upper left", handlelength=2.0)

    # ---- painel (b) -------------------------------------------------------
    p = instancia_sintetica(64, 0.25, rng)
    at = list(p)
    Qmax = 0.35 * sum(p[w].qg_max_lucro for w in at)
    t0 = time.perf_counter()
    for _ in range(50):
        resolve_kkt(p, at, Qmax, economico=True)
    unidade = (time.perf_counter() - t0) / 50.0

    ns = np.arange(8, 141, 4)
    for frac, cor, traco, rot in ((0.10, AZUL, (0, (4, 1.6)), "10%"),
                                  (0.25, LARANJA, "-", "25%"),
                                  (0.50, VERDE, (0, (1.6, 1.4)), "50%")):
        m = np.maximum(1, np.round(frac * ns)).astype(int)
        seg = (2.0 ** m) * unidade
        ax2.plot(ns, seg, color=cor, linestyle=traco, linewidth=1.6,
                 label=f"{rot} dos poços são tipo c ou d", zorder=4)

    ax2.axhline(3600, color=TINTA, linewidth=0.8, linestyle=(0, (3, 2)),
                zorder=3)
    ax2.annotate("uma hora", (ns[-1], 3600), textcoords="offset points",
                 xytext=(-3, 5), fontsize=7.2, color=TINTA, ha="right")
    ax2.axvline(8, color=CINZA, linewidth=0.9, zorder=2)
    ax2.annotate("os 8 poços do campo JD", (8, 2e11),
                 textcoords="offset points", xytext=(6, 0), fontsize=7.2,
                 color=TINTA, va="top", ha="left")

    ax2.set_yscale("log")
    ax2.set_xlabel("número de poços")
    ax2.set_ylabel("tempo da enumeração (s)")
    ax2.set_ylim(1e-6, 1e12)
    ax2.legend(frameon=False, loc="lower right", handlelength=2.0)

    for ax, letra in ((ax1, "(a)"), (ax2, "(b)")):
        _limpa(ax)
        ax.text(0.015, 1.04, letra, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=TINTA)

    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def gerar_todas(pocos: dict) -> None:
    print("\n  gerando figuras")
    for f in (fig_curvas(pocos), fig_alocacao(pocos), fig_escala()):
        print(f"    {f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from gaslift import monta_pocos
    gerar_todas(monta_pocos())


# ===========================================================================
#  FIGURA DOS QUATRO TIPOS DE CURVA ECONOMICA
# ===========================================================================

ROXO = "#9333EA"   # quarta cor da paleta categorica validada

def fig_tipos(pocos: dict, caminho: str = "fig_tipos.png") -> str:
    """
    Os quatro formatos de curva economica descritos em [5], ilustrados por
    pocos reais do campo JD.

    O que separa os grupos e o sinal de gama, que e o lucro do poco sem gas
    nenhum. Os tipos a e b tem gama positivo e ja sao rentaveis parados. Os
    tipos c e d tem gama negativo e so se pagam acima de uma injecao minima.
    Dentro de cada grupo, a distincao e de intensidade de resposta ao gas.
    """
    _estilo()
    fig, ax = plt.subplots(figsize=(5.0, 3.1))

    sel = [("W4", "a", AZUL,    (0, (5, 1.8))),
           ("W6", "b", LARANJA, "-"),
           ("W7", "c", VERDE,   (0, (1.8, 1.5))),
           ("W8", "d", ROXO,    (0, (5, 1.6, 1.2, 1.6)))]

    qg = np.linspace(0.0, 2.9, 400)
    for nome, tipo, cor, traco in sel:
        p = pocos[nome]
        y = [p.lucro(float(x)) for x in qg]
        ax.plot(qg, y, color=cor, linestyle=traco, linewidth=1.7, zorder=3,
                label=f"tipo {tipo}  ({nome})")
        ax.plot(p.qg_max_lucro, p.lucro_max, "o", color=cor, markersize=4.0,
                markeredgecolor="white", markeredgewidth=0.7, zorder=5)
        qmin = p.qg_min_economico
        if qmin is not None:
            ax.plot(qmin, 0.0, "v", color=cor, markersize=5.0,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=6)

    ax.axhline(0.0, color=TINTA, linewidth=0.8, zorder=2)

    # anotacoes que nomeiam os pontos notaveis uma unica vez
    pa = pocos["W4"]
    ax.annotate(r"$\gamma_i>0$: lucra sem gás", xy=(0.02, pa.gama),
                xytext=(0.22, pa.gama + 780), fontsize=7.2, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=CINZA, linewidth=0.6,
                                shrinkA=1, shrinkB=2))
    pd = pocos["W8"]
    ax.annotate(r"$q_{g,\min}$: limiar de rentabilidade",
                xy=(pd.qg_min_economico, 0.0), xytext=(0.62, -1550),
                fontsize=7.2, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=CINZA, linewidth=0.6,
                                shrinkA=1, shrinkB=3))
    pb = pocos["W6"]
    ax.annotate(r"$q_{g}^{\max}$: ótimo do poço isolado",
                xy=(pb.qg_max_lucro, pb.lucro_max),
                xytext=(1.02, pb.lucro_max + 1180), fontsize=7.2, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=CINZA, linewidth=0.6,
                                shrinkA=1, shrinkB=3))

    ax.set_xlabel(r"gás injetado $q_{g,i}$  (10$^4$ m$^3$/d)")
    ax.set_ylabel(r"lucro diário $y_i$  (US\$/d)")
    ax.set_xlim(0, 2.9)
    ax.set_ylim(-1850, 3850)
    _limpa(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=2.4, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# ===========================================================================
#  FIGURA DE RESULTADOS (substitui a versao em curvas)
# ===========================================================================

def fig_resultado(pocos: dict, caminho: str = "fig_resultado.png") -> str:
    """
    (a) o que cada poco queria contra o que recebeu, com o gas restrito
    (b) como a reparticao muda conforme o gas disponivel cresce

    Em (a) os pocos aparecem ordenados pela receita marginal em vazao nula,
    que e o criterio que de fato decide a prioridade. Fica visivel que o poco
    de maior apetite isolado pode ser o ultimo da fila.
    """
    from gaslift import ESQUEMAS, resolve_penalidade

    _estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.58),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    ativos = ESQUEMAS[1]

    # ---- painel (a): queria contra recebeu, em Qmax = 8 -------------------
    Qref = 8.0
    q = dict(zip(ativos, resolve_penalidade(pocos, ativos, Qref, economico=True)[0]))
    ordem = sorted(ativos, key=lambda w: -pocos[w].beta)
    y = np.arange(len(ordem))
    quer = [pocos[w].qg_max_lucro for w in ordem]
    teve = [q[w] for w in ordem]

    ax1.barh(y + 0.19, quer, height=0.34, color=CINZA, alpha=0.55,
             label="ótimo isolado", zorder=3)
    ax1.barh(y - 0.19, teve, height=0.34, color=AZUL,
             label=r"alocado com $Q_{\max}=8$", zorder=4)
    for k, w in enumerate(ordem):
        pct = teve[k] / quer[k] * 100.0
        ax1.annotate(f"{pct:.0f}%", (teve[k], y[k] - 0.19),
                     textcoords="offset points", xytext=(4, 0), fontsize=7.0,
                     va="center", color=TINTA)
    ax1.set_yticks(y)
    ax1.set_yticklabels(ordem)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"gás alocado ao poço  (10$^4$ m$^3$/d)")
    ax1.set_xlim(0, 3.05)
    ax1.grid(True, axis="x", zorder=0); ax1.grid(False, axis="y")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=7.2, handlelength=1.4, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.20),
               columnspacing=1.8)
    ax1.text(0.015, 1.05, "(a)", transform=ax1.transAxes, fontsize=8.5,
             fontweight="bold", color=TINTA)
    ax1.set_ylabel("poço, em ordem de receita marginal", fontsize=8.0)

    # ---- painel (b): composicao da alocacao por nivel de gas --------------
    niveis = [2.0, 4.0, 8.0, 12.0]
    rampa = plt.cm.Blues(np.linspace(0.90, 0.30, len(ordem)))
    base = np.zeros(len(niveis))
    aloc = {w: [] for w in ordem}
    for Q in niveis:
        qq = dict(zip(ativos, resolve_penalidade(pocos, ativos, Q, economico=True)[0]))
        for w in ordem:
            aloc[w].append(qq[w])
    x = np.arange(len(niveis))
    for k, w in enumerate(ordem):
        v = np.array(aloc[w])
        ax2.bar(x, v, bottom=base, width=0.62, color=rampa[k],
                edgecolor="white", linewidth=0.8, zorder=3)
        for j in range(len(niveis)):          # rotula os segmentos maiores
            if v[j] >= 1.15:
                ax2.annotate(w, (x[j], base[j] + v[j] / 2), ha="center",
                             va="center", fontsize=6.4, zorder=5,
                             color="white" if k < 4 else TINTA,
                             fontweight="bold")
        base += v
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(n)}" for n in niveis])
    ax2.set_xlabel(r"gás disponível no bloco $Q_{\max}$  (10$^4$ m$^3$/d)")
    ax2.set_ylabel(r"gás alocado  (10$^4$ m$^3$/d)")
    ax2.grid(True, axis="y", zorder=0); ax2.grid(False, axis="x")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.text(0.015, 1.05, "(b)", transform=ax2.transAxes, fontsize=8.5,
             fontweight="bold", color=TINTA)
    ax2.annotate("tom mais escuro,\nmaior prioridade", (x[0] - 0.40, 8.6),
                 fontsize=7.0, color=TINTA, style="italic", ha="left",
                 va="top")

    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho
