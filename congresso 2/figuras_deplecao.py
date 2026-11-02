#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura da camada de depletacao.

    fig_deplecao.png   (a) trajetoria da injecao otima ao longo do horizonte
                       (b) fator de capacidade contra o limiar de abandono

Uso:
    python deplecao.py --figuras
    python figuras_deplecao.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from figuras import AZUL, LARANJA, VERDE, TINTA, CINZA, _estilo, _limpa
from gaslift import TABELA_1, P4_INJ, fator_economico, monta_pocos, POCOS
import deplecao as dp

ROXO = "#9333EA"


def fig_deplecao(pocos, reservas, qg, N, qg_campo=None) -> None:
    """
    Dois paineis.

    (a) A injecao otima de cada poco ano a ano. Tres trajetorias contam a
        historia e ficam em cor; as demais ficam em cinza, para nao competir.
        O W4, o poco mais produtivo do bloco, e o que mais perde gas, porque e
        o que esgota primeiro. O W5, que o modelo estatico descarta, e o que
        mais ganha, porque e o que tem mais reserva por unidade de producao.

    (b) O fator de capacidade remanescente de cada poco contra o proprio
        limiar de abandono d*. O cruzamento do W4 dentro do horizonte e o
        momento em que a sua injecao se anula no painel (a).
    """
    _estilo()
    m, T = qg.shape
    anos = np.arange(1, T + 1)
    Nmax = np.array([reservas[w] for w in POCOS])

    destaque = {"W4": LARANJA, "W5": AZUL, "W8": VERDE}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75))

    # ---- (a) trajetoria da injecao -----------------------------------------
    # linha cheia, a politica que antecipa; tracejada, o metodo de campo
    for k, nome in enumerate(POCOS):
        if nome in destaque:
            ax1.plot(anos, qg[k], color=destaque[nome], lw=1.7, zorder=3,
                     marker="o", ms=2.6, mew=0)
            if qg_campo is not None:
                ax1.plot(anos, qg_campo[k], color=destaque[nome], lw=1.3,
                         ls=(0, (4, 2)), zorder=3, alpha=0.9)
        else:
            ax1.plot(anos, qg[k], color=CINZA, lw=0.9, zorder=2, alpha=0.75)
    _limpa(ax1)
    ax1.set_xlabel("ano")
    ax1.set_ylabel(r"$q_{g}$  [$10^4$ m$^3$/d]")
    ax1.set_xlim(0.5, T + 0.5)
    ax1.set_ylim(0, 1.95)
    ax1.xaxis.set_major_locator(MultipleLocator(3))

    # W8 e W5 ficam isolados no fim; W4 termina colado no eixo, e por isso
    # recebe o rotulo no inicio da propria curva.
    for nome in ("W8", "W5"):
        k = POCOS.index(nome)
        ax1.annotate(nome, xy=(T, qg[k, -1]), xytext=(-1, 6),
                     textcoords="offset points", color=destaque[nome],
                     fontsize=8.0, fontweight="bold", ha="right")
    k4 = POCOS.index("W4")
    ax1.annotate("W4", xy=(1, qg[k4, 0]), xytext=(3, 7),
                 textcoords="offset points", color=LARANJA, fontsize=8.0,
                 fontweight="bold", ha="left")
    ax1.annotate("os cinco demais", xy=(5.5, 1.02), color=CINZA, fontsize=7.4,
                 ha="center")
    # o contraste que importa, o W5 no inicio do horizonte
    ax1.annotate("o campo só liga o W5 no ano 5", xy=(3.2, 0.015),
                 xytext=(5.2, 0.055), color=AZUL, fontsize=7.0, ha="left",
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=AZUL, lw=0.7,
                                 shrinkA=4, shrinkB=3))
    if qg_campo is not None:
        from matplotlib.lines import Line2D
        ax1.legend(handles=[
            Line2D([0], [0], color=TINTA, lw=1.5, label="antecipa"),
            Line2D([0], [0], color=TINTA, lw=1.2, ls=(0, (4, 2)),
                   label="campo")],
            loc="upper center", bbox_to_anchor=(0.52, 1.0), frameon=False,
            fontsize=7.2, handlelength=2.2, ncol=2, columnspacing=1.1,
            handletextpad=0.5)
    ax1.set_title("(a) injecao ano a ano", pad=5)

    # ---- (b) capacidade contra o limiar -------------------------------------
    t_eixo = np.arange(0, T + 1)
    for k, nome in enumerate(POCOS):
        d = 1.0 - N[k] / Nmax[k]
        cor = destaque.get(nome, CINZA)
        lw = 1.7 if nome in destaque else 0.9
        ax2.plot(t_eixo, d, color=cor, lw=lw,
                 zorder=3 if nome in destaque else 2,
                 alpha=1.0 if nome in destaque else 0.75)
    # limiares d* dos pocos em destaque
    for nome, cor in destaque.items():
        A, B, C, Fw, GOR = TABELA_1[nome]
        dstar = P4_INJ / (fator_economico(Fw, GOR) * B)
        ax2.plot([0, T], [dstar, dstar], color=cor, lw=0.8, ls=(0, (3, 2)),
                 zorder=2, alpha=0.9)
    _limpa(ax2)
    ax2.set_xlabel("ano")
    ax2.set_ylabel(r"fator de capacidade  $d$")
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1.04)
    ax2.xaxis.set_major_locator(MultipleLocator(3))

    k4 = POCOS.index("W4")
    d4 = 1.0 - N[k4] / Nmax[k4]
    A, B, C, Fw, GOR = TABELA_1["W4"]
    d4star = P4_INJ / (fator_economico(Fw, GOR) * B)
    # instante do cruzamento, por interpolacao linear
    idx = int(np.argmax(d4 < d4star))
    if idx > 0:
        t0, t1 = idx - 1, idx
        frac = (d4[t0] - d4star) / (d4[t0] - d4[t1])
        t_cruz = t0 + frac
        ax2.plot([t_cruz], [d4star], marker="o", ms=4.5, color=LARANJA,
                 mew=0, zorder=5)
        ax2.annotate(r"W4 cruza o seu $d^{*}$", xy=(t_cruz, d4star),
                     xytext=(5.6, 0.045), color=LARANJA, fontsize=7.3,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=LARANJA, lw=0.7,
                                     shrinkA=3, shrinkB=3,
                                     connectionstyle="arc3,rad=-0.2"))
    else:
        # o W4 nao chega a cruzar, mas termina sobre o proprio limiar
        ax2.plot([T], [d4[-1]], marker="o", ms=4.5, color=LARANJA, mew=0,
                 zorder=5)
        ax2.annotate(r"W4 alcança o seu $d^{*}$", xy=(T, d4[-1]),
                     xytext=(5.4, 0.055), color=LARANJA, fontsize=7.3,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=LARANJA, lw=0.7,
                                     shrinkA=3, shrinkB=4,
                                     connectionstyle="arc3,rad=-0.25"))
    # W5 e rotulado no fim, onde esta isolado; W8 e W4 terminam colados, e por
    # isso sao rotulados sobre a propria curva, em anos distintos.
    k5 = POCOS.index("W5")
    d5 = 1.0 - N[k5] / Nmax[k5]
    ax2.annotate("W5", xy=(T, d5[-1]), xytext=(-1, 6),
                 textcoords="offset points", color=AZUL, fontsize=8.0,
                 fontweight="bold", ha="right")
    k8 = POCOS.index("W8")
    d8 = 1.0 - N[k8] / Nmax[k8]
    ax2.annotate("W8", xy=(5, d8[5]), xytext=(-13, -2),
                 textcoords="offset points", color=VERDE, fontsize=8.0,
                 fontweight="bold", ha="right")
    ax2.annotate("W4", xy=(9, d4[9]), xytext=(-12, -3),
                 textcoords="offset points", color=LARANJA, fontsize=8.0,
                 fontweight="bold", ha="right")
    ax2.set_title(r"(b) capacidade e limiar de abandono", pad=5)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.savefig("fig_deplecao.png")
    plt.close(fig)
    print("  fig_deplecao.png")


def fig_curvas(pocos, poco: str = "W4") -> None:
    """
    O que a depleção faz com as duas curvas de um mesmo poço.

    (a) A curva de desempenho encolhe verticalmente, mantendo a forma, porque o
        fator de capacidade d multiplica os tres coeficientes. O ponto de maxima
        producao, -B/2A, nao se move.

    (b) A curva economica encolhe de outro jeito, porque o custo de pressurizacao
        p_4 q_g nao e multiplicado por d. O ponto de maximo lucro recua em
        direcao a origem, e abaixo de d* a curva ja nasce descendo, de modo que
        nenhuma injecao se paga.
    """
    _estilo()
    A, B, C, Fw, GOR = TABELA_1[poco]
    K = fator_economico(Fw, GOR)
    dstar = P4_INJ / (K * B)
    niveis = [1.00, 0.75, 0.50, 0.25]
    cores = plt.cm.Blues(np.linspace(0.88, 0.34, len(niveis)))

    qg = np.linspace(0.0, 2.6, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.7))

    # ---- (a) curva de desempenho -------------------------------------------
    for d, cor in zip(niveis, cores):
        ax1.plot(qg, d * (A * qg ** 2 + B * qg + C), color=cor, lw=1.6,
                 zorder=3)
    qtec = -B / (2.0 * A)
    ax1.axvline(qtec, color=TINTA, lw=0.8, ls=(0, (3, 2)), zorder=2)
    _limpa(ax1)
    ax1.set_xlabel(r"$q_{g}$  [$10^4$ m$^3$/d]")
    ax1.set_ylabel(r"$q_{o}$  [m$^3$/d]")
    ax1.set_xlim(0, 2.6)
    ax1.set_ylim(0, 12.6)
    ax1.annotate(r"$-B/2A$ não se move", xy=(qtec, 11.9), xytext=(5, 0),
                 textcoords="offset points", color=TINTA, fontsize=7.3,
                 ha="left", va="center")
    # rotulo pousado sobre a propria curva, no trecho em que elas correm
    # paralelas e bem separadas
    xr = 0.62
    for d, cor in zip(niveis, cores):
        ax1.annotate((r"$d=%.2f$" % d).replace(".", ","),
                     xy=(xr, d * (A * xr ** 2 + B * xr + C)),
                     xytext=(0, 7), textcoords="offset points", color=cor,
                     fontsize=7.2, ha="center", va="bottom")
    ax1.set_title("(a) curva de desempenho", pad=5)

    # ---- (b) curva economica ------------------------------------------------
    for d, cor in zip(niveis, cores):
        y = K * d * (A * qg ** 2 + B * qg + C) - P4_INJ * qg
        ax2.plot(qg, y, color=cor, lw=1.6, zorder=3)
        qeco = -(K * B * d - P4_INJ) / (2.0 * K * A * d)
        if qeco > 0:
            yeco = K * d * (A * qeco ** 2 + B * qeco + C) - P4_INJ * qeco
            ax2.plot([qeco], [yeco], marker="o", ms=4.2, color=cor, mew=0,
                     zorder=5)
    # a curva no limiar d*, em que o maximo ja esta na origem
    ylim = K * dstar * (A * qg ** 2 + B * qg + C) - P4_INJ * qg
    ax2.plot(qg, ylim, color=LARANJA, lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax2.axhline(0.0, color=CINZA, lw=0.8, zorder=1)
    _limpa(ax2)
    ax2.set_xlabel(r"$q_{g}$  [$10^4$ m$^3$/d]")
    ax2.set_ylabel(r"$y$  [US\$/d]")
    ax2.set_xlim(0, 2.6)
    ax2.set_ylim(-700, 3700)
    ax2.annotate("máximos recuam", xy=(0.40, 480), xytext=(1.08, 1180),
                 color=TINTA, fontsize=7.3, ha="left",
                 arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.7,
                                 shrinkA=2, shrinkB=4,
                                 connectionstyle="arc3,rad=0.25"))
    ax2.annotate((r"$d^{*}=%.3f$" % dstar).replace(".", ","), xy=(1.30, -340),
                 color=LARANJA, fontsize=7.3, ha="left")
    ax2.set_title("(b) curva econômica", pad=5)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.savefig("fig_curvas_deplecao.png")
    plt.close(fig)
    print("  fig_curvas_deplecao.png")


def gerar(pocos=None) -> None:
    if pocos is None:
        pocos = monta_pocos(usar_tabela_3=True)
    print("Figuras geradas:")
    fig_curvas(pocos)
    reservas = dp.calibra_reservas(pocos)
    qg, N = dp.resolve_multiperiodo(pocos, reservas)[:2]
    fig_deplecao(pocos, reservas, qg, N)


if __name__ == "__main__":
    gerar()
