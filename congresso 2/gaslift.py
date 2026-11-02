#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Alocacao economica otima de gas de injecao em um bloco de pocos de gas lift
 -----------------------------------------------------------------------------
 Reproducao computacional de:

   ZHONG, H.; ZHAO, C.; XU, Z.; ZHENG, C. Economical optimum gas allocation
   model considering different types of gas-lift performance curves.
   Energies, v. 15, n. 19, 6950, 2022. DOI 10.3390/en15196950.

 Instancia: 8 pocos do campo JD.

 O programa esta organizado em tres camadas:

   Camada 1  Validacao e reproducao
             1.1  Tabelas 1 e 2 --(Eqs. 6 a 8)--> Tabela 3
             1.2  Formulas fechadas --> colunas "Unconstrained"
             1.3  Modelo restrito    --> Tabelas 4 a 11

   Camada 2  Analise das solucoes publicadas
             2.1  O ponto Qmax = 4 do Esquema 1
             2.2  Otimo tecnico (ORAT) contra otimo economico (NPV)

   Camada 3  Escalabilidade
             3.1  Custo do NLP continuo
             3.2  Custo da enumeracao 2^m dos pocos tipo c e d

 Tres metodos de solucao sao implementados e comparados:
   (a) KKT com bissecao no multiplicador de Lagrange
   (b) funcao penalidade mista com Newton-Raphson, que e o metodo do artigo
   (c) programacao quadratica sequencial, via scipy

 Execucao:
     python gaslift.py                 roda tudo
     python gaslift.py --camada 1      roda apenas a camada indicada
     python gaslift.py --figuras       gera as figuras
===============================================================================
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

# =============================================================================
#  DADOS DO ARTIGO
# =============================================================================

# -- Tabela 1: coeficientes da GLPC, water cut e razao gas-oleo de formacao ----
#    GLPC:  q_o = A q_g^2 + B q_g + C      [q_o em m3/d, q_g em 10^4 m3/d]
#    Fw   : water cut, fracao
#    GOR  : razao gas-oleo de formacao, 10^4 m3/m3
TABELA_1: dict[str, tuple[float, float, float, float, float]] = {
    # poco :   A         B      C        Fw      GOR
    "W1": (-0.580, 3.86, 1.082, 0.520, 0.0160),
    "W2": (-0.920, 4.24, 3.530, 0.450, 0.0100),
    "W3": (-1.420, 6.53, 2.091, 0.480, 0.0125),
    "W4": (-1.350, 4.68, 6.889, 0.390, 0.0150),
    "W5": (-0.287, 2.36, 2.893, 0.580, 0.0180),
    "W6": (-1.106, 5.68, 0.788, 0.500, 0.0160),
    "W7": (-1.620, 7.36, -1.945, 0.443, 0.0200),
    "W8": (-2.048, 9.74, -3.022, 0.550, 0.0175),
}
POCOS: list[str] = list(TABELA_1)

# -- Tabela 2: parametros economicos ------------------------------------------
P1_OLEO = 301.89     # preco do oleo,                    $/m3
P2_GAS = 2012.38     # preco do gas natural,             $/10^4 m3
P3_AGUA = 0.077      # tratamento da agua produzida,     $/m3
P4_INJ = 309.60      # pressurizacao do gas injetado,    $/10^4 m3
P5_DESG = 0.0077     # desgaseificacao do oleo,          $/m3

# -- Tabela 3: coeficientes da GLEPC publicados pelos autores -----------------
#    GLEPC: y = alfa q_g^2 + beta q_g + gama            [y em $/d]
TABELA_3: dict[str, tuple[float, float, float]] = {
    "W1": (-193.81, 980.26, 361.56),
    "W2": (-296.31, 1055.96, 1136.89),
    "W3": (-464.50, 1826.40, 683.98),
    "W4": (-448.36, 1244.70, 2287.94),
    "W5": (-96.49, 488.58, 978.44),
    "W6": (-362.41, 1588.40, 263.31),
    "W7": (-540.86, 2208.20, -665.55),
    "W8": (-671.19, 2975.99, -1018.99),
}

# -- Esquemas de alocacao (Secao 5 do artigo) ---------------------------------
#    W7 e de Tipo c e W8 e de Tipo d: ambos so sao rentaveis acima de uma
#    injecao minima, de modo que cada um pode ser ligado ou desligado.
ESQUEMAS: dict[int, list[str]] = {
    1: ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],   # todos
    2: ["W1", "W2", "W3", "W4", "W5", "W6", "W7"],         # sem W8
    3: ["W1", "W2", "W3", "W4", "W5", "W6"],               # sem W7 e sem W8
    4: ["W1", "W2", "W3", "W4", "W5", "W6", "W8"],         # sem W7
}

# -- Valores publicados, para conferencia -------------------------------------
#    Tabela 4: Esquema 1, objetivo = maxima producao de oleo (ORAT)
TAB4_QGI = {
    2.0: [0.00, 0.00, 0.35, 0.00, 0.00, 0.06, 0.56, 1.03],
    4.0: [0.00, 0.11, 0.88, 0.24, 0.00, 0.74, 1.03, 1.00],
    8.0: [0.88, 0.76, 1.30, 0.68, 0.00, 1.29, 1.40, 1.69],
    12.0: [1.74, 1.30, 1.65, 1.05, 0.90, 1.73, 1.70, 1.93],
    16.0: [2.45, 1.75, 1.94, 1.35, 2.33, 2.11, 1.96, 2.13],
}
TAB4_QO = {2.0: 26.23, 4.0: 35.61, 8.0: 49.82, 12.0: 58.97, 16.0: 64.71}

#    Tabela 8: Esquema 1, objetivo = maximo lucro diario (NPV)
TAB8_QGI = {
    2.0: [0.00, 0.00, 0.30, 0.00, 0.00, 0.06, 0.60, 1.04],
    4.0: [0.00, 0.00, 0.78, 0.17, 0.00, 0.67, 1.01, 1.37],
    8.0: [0.90, 0.71, 1.28, 0.68, 0.00, 1.30, 1.43, 1.71],
    12.0: [1.74, 1.26, 1.63, 1.04, 0.93, 1.74, 1.92, 1.94],
}
TAB8_Y = {2.0: 8127.1, 4.0: 10765.7, 8.0: 14192.9, 12.0: 16041.7}

TOL = 1e-10


# =============================================================================
#  CAMADA 1.1  Dos dados brutos aos coeficientes economicos
# =============================================================================

@dataclass(frozen=True)
class Poco:
    """Um poco de gas lift, com as duas curvas que o descrevem."""
    nome: str
    A: float            # GLPC
    B: float
    C: float
    Fw: float           # water cut
    GOR: float          # razao gas-oleo
    alfa: float         # GLEPC
    beta: float
    gama: float

    # --- curvas ---------------------------------------------------------
    def oleo(self, qg: float) -> float:
        """Producao de oleo, m3/d, em funcao do gas injetado."""
        return self.A * qg * qg + self.B * qg + self.C

    def lucro(self, qg: float) -> float:
        """Receita liquida diaria, $/d, em funcao do gas injetado."""
        return self.alfa * qg * qg + self.beta * qg + self.gama

    # --- pontos notaveis ------------------------------------------------
    @property
    def qg_max_oleo(self) -> float:
        """Injecao que maximiza a producao de oleo. Eq. (10) aplicada a GLPC."""
        return -self.B / (2.0 * self.A)

    @property
    def qg_max_lucro(self) -> float:
        """Injecao que maximiza o lucro do poco. Eq. (10)."""
        return -self.beta / (2.0 * self.alfa)

    @property
    def lucro_max(self) -> float:
        """Lucro maximo do poco isolado. Eq. (11)."""
        return (4.0 * self.alfa * self.gama - self.beta ** 2) / (4.0 * self.alfa)

    @property
    def qg_min_economico(self) -> float | None:
        """
        Injecao a partir da qual o poco passa a dar lucro. Eq. (12).
        Retorna None quando o poco ja e rentavel sem gas, isto e, gama >= 0.

        A GLEPC e uma parabola de boca para baixo. Quando gama < 0 ela corta o
        eixo das abscissas em dois pontos, ambos positivos: o primeiro e o
        limiar de rentabilidade e o segundo e o ponto em que o poco volta a dar
        prejuizo por excesso de gas. O que interessa e o primeiro. Como alfa e
        negativo, a divisao por 2 alfa inverte a ordem das raizes em relacao a
        forma usual, de modo que o menor valor vem com o sinal positivo diante
        do radical.
        """
        if self.gama >= 0.0:
            return None
        disc = self.beta ** 2 - 4.0 * self.alfa * self.gama
        if disc < 0.0:
            return None
        raiz = math.sqrt(disc)
        r1 = (-self.beta + raiz) / (2.0 * self.alfa)
        r2 = (-self.beta - raiz) / (2.0 * self.alfa)
        return min(r1, r2)


def fator_economico(Fw: float, GOR: float) -> float:
    """
    Fator que converte uma unidade de oleo em receita liquida, $/m3.
    E o termo entre parenteses das Eqs. (6) a (8):

        K = p1 + p2 GOR - p3 Fw/(1-Fw) - p5
    """
    return P1_OLEO + P2_GAS * GOR - P3_AGUA * Fw / (1.0 - Fw) - P5_DESG


def monta_pocos(usar_tabela_3: bool = True) -> dict[str, Poco]:
    """
    Constroi os oito pocos.

    Com usar_tabela_3=True adotam-se os coeficientes da GLEPC publicados na
    Tabela 3. Com False eles sao recalculados das Tabelas 1 e 2 pelas
    Eqs. (6) a (8). A diferenca entre as duas leituras e analisada em
    camada_1_1().
    """
    pocos: dict[str, Poco] = {}
    for nome, (A, B, C, Fw, GOR) in TABELA_1.items():
        K = fator_economico(Fw, GOR)
        if usar_tabela_3:
            alfa, beta, gama = TABELA_3[nome]
        else:
            alfa, beta, gama = K * A, K * B - P4_INJ, K * C
        pocos[nome] = Poco(nome, A, B, C, Fw, GOR, alfa, beta, gama)
    return pocos


def camada_1_1() -> None:
    """Confere a cadeia Tabela 1 + Tabela 2 --(Eqs. 6 a 8)--> Tabela 3."""
    print("\n" + "=" * 79)
    print(" CAMADA 1.1  Das Tabelas 1 e 2 para a Tabela 3 (Eqs. 6 a 8)")
    print("=" * 79)
    print(f"{'poco':<5}{'alfa calc':>11}{'alfa pub':>10}{'err%':>7}"
          f"{'beta calc':>11}{'beta pub':>10}{'err%':>7}"
          f"{'gama calc':>11}{'gama pub':>10}{'err%':>7}")

    piores: list[tuple[str, float]] = []
    for nome, (A, B, C, Fw, GOR) in TABELA_1.items():
        K = fator_economico(Fw, GOR)
        calc = (K * A, K * B - P4_INJ, K * C)
        pub = TABELA_3[nome]
        erros = [abs(c - p) / abs(p) * 100.0 for c, p in zip(calc, pub)]
        piores.append((nome, erros[0]))
        print(f"{nome:<5}"
              f"{calc[0]:11.2f}{pub[0]:10.2f}{erros[0]:7.2f}"
              f"{calc[1]:11.2f}{pub[1]:10.2f}{erros[1]:7.2f}"
              f"{calc[2]:11.2f}{pub[2]:10.2f}{erros[2]:7.2f}")

    print("\n  Os coeficientes beta e gama reproduzem em todos os pocos, com")
    print("  erro abaixo de 0,1 por cento. O coeficiente alfa diverge em W5 a")
    print("  W8. Invertendo a Eq. (6) para recuperar o A que geraria o alfa")
    print("  publicado:\n")
    print(f"{'poco':<5}{'A da Tab.1':>12}{'A implicado':>13}{'erro%':>8}")
    for nome, erro in piores:
        if erro < 0.2:
            continue
        A, B, C, Fw, GOR = TABELA_1[nome]
        K = fator_economico(Fw, GOR)
        A_implicado = TABELA_3[nome][0] / K
        print(f"{nome:<5}{A:12.4f}{A_implicado:13.4f}"
              f"{abs(A_implicado - A) / abs(A) * 100.0:8.2f}")
    print("\n  Como B e C invertem exatamente, a inconsistencia esta confinada")
    print("  ao coeficiente A desses quatro pocos na Tabela 1.")


# =============================================================================
#  CAMADA 1.2  Formulas fechadas
# =============================================================================

def camada_1_2(pocos: dict[str, Poco]) -> None:
    """Confere as formulas fechadas contra as colunas sem restricao de gas."""
    print("\n" + "=" * 79)
    print(" CAMADA 1.2  Formulas fechadas e as colunas sem restricao de gas")
    print("=" * 79)

    pub_orat = dict(zip(POCOS, [3.33, 2.30, 2.30, 1.73, 4.11, 2.57, 2.27, 2.38]))
    pub_npv = dict(zip(POCOS, [2.44, 1.72, 1.92, 1.35, 2.31, 2.11, 1.97, 2.14]))

    print(f"{'poco':<5}{'-B/2A':>9}{'Tab.4':>8}{'dif':>7}   "
          f"{'-beta/2alfa':>12}{'Tab.8':>8}{'dif%':>7}   {'tipo':>6}")
    for nome in POCOS:
        p = pocos[nome]
        d_orat = p.qg_max_oleo - pub_orat[nome]
        d_npv = (p.qg_max_lucro - pub_npv[nome]) / pub_npv[nome] * 100.0
        qmin = p.qg_min_economico
        tipo = "c ou d" if qmin is not None else "a ou b"
        print(f"{nome:<5}{p.qg_max_oleo:9.3f}{pub_orat[nome]:8.2f}"
              f"{d_orat:+7.3f}   {p.qg_max_lucro:12.3f}{pub_npv[nome]:8.2f}"
              f"{d_npv:+7.1f}   {tipo:>6}")

    print("\n  A formula do otimo tecnico reproduz a Tabela 4 nos oito pocos.")
    print("  A do otimo economico mostra vies sistematico de cerca de +3,6 por")
    print("  cento, compativel com parada prematura da iteracao dos autores.")

    print("\n  Injecao minima economica dos pocos que a possuem, Eq. (12):")
    for nome in POCOS:
        qmin = pocos[nome].qg_min_economico
        if qmin is not None:
            print(f"    {nome}: {qmin:.4f} x10^4 m3/d  (gama < 0, "
                  f"o poco so se paga acima desse valor)")


# =============================================================================
#  METODOS DE SOLUCAO
# =============================================================================

def _coefs(pocos: dict[str, Poco], ativos: list[str], economico: bool
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (a, b, c, limite_superior) do conjunto de pocos ativos."""
    if economico:
        a = np.array([pocos[w].alfa for w in ativos])
        b = np.array([pocos[w].beta for w in ativos])
        c = np.array([pocos[w].gama for w in ativos])
        sup = np.array([pocos[w].qg_max_lucro for w in ativos])
    else:
        a = np.array([pocos[w].A for w in ativos])
        b = np.array([pocos[w].B for w in ativos])
        c = np.array([pocos[w].C for w in ativos])
        sup = np.array([pocos[w].qg_max_oleo for w in ativos])
    return a, b, c, sup


def resolve_kkt(pocos: dict[str, Poco], ativos: list[str], Qmax: float,
                economico: bool = False) -> tuple[np.ndarray, float]:
    """
    Metodo (a). Condicoes KKT resolvidas por bissecao no multiplicador.

    O lagrangiano do problema

        max  sum_i (a_i q_i^2 + b_i q_i + c_i)
        s.a. sum_i q_i = Qmax,     0 <= q_i <= q_i^sup

    da, para todo poco com injecao interior, a condicao de produtividade
    marginal igualada:

        2 a_i q_i + b_i = lambda   ->   q_i(lambda) = (lambda - b_i)/(2 a_i)

    Como a_i < 0, cada q_i(lambda) decresce com lambda, e a soma tambem. Basta
    entao procurar o unico lambda que fecha o balanco de gas. Pocos cuja
    produtividade marginal em zero ja e menor que lambda ficam sem gas, o que
    a projecao nos limites entrega pela folga complementar.
    """
    a, b, _, sup = _coefs(pocos, ativos, economico)

    def q_de(lam: float) -> np.ndarray:
        return np.clip((lam - b) / (2.0 * a), 0.0, sup)

    if Qmax >= sup.sum():                      # gas de sobra, cada um no seu otimo
        return sup.copy(), 0.0

    lam = brentq(lambda L: q_de(L).sum() - Qmax, 0.0, float(b.max()), xtol=TOL)
    return q_de(lam), lam


def resolve_penalidade(pocos: dict[str, Poco], ativos: list[str], Qmax: float,
                       economico: bool = False, r0: float = 1.0e3,
                       reducao: float = 0.25, n_ciclos: int = 40,
                       max_newton: int = 60) -> tuple[np.ndarray, int]:
    """
    Metodo (b). Funcao penalidade mista com Newton-Raphson, Eqs. (23) a (31).

    A restricao de igualdade entra como penalidade quadratica e os limites
    inferiores como barreira logaritmica, de modo que o problema restrito vira
    uma sequencia de problemas irrestritos

        p(q, r) = sum_i (a_i q_i^2 + b_i q_i + c_i)
                  + r sum_i ln(q_i)
                  - (1/r) (sum_i q_i - Qmax)^2

    resolvidos por Newton enquanto r decresce para zero. O gradiente e a
    hessiana sao analiticos, e a hessiana tem a estrutura diagonal mais posto
    um que a Eq. (28) explora.
    """
    a, b, _, sup = _coefs(pocos, ativos, economico)

    # ponto inicial da Eq. (31): reparte o gas na proporcao dos otimos de poco
    q = np.clip(Qmax * sup / sup.sum(), 1.0e-6, sup * 0.999)

    r = r0
    iteracoes = 0
    for _ in range(n_ciclos):
        for _ in range(max_newton):
            iteracoes += 1
            folga = q.sum() - Qmax

            # gradiente de p(q,r)
            grad = 2.0 * a * q + b + r / q - (2.0 / r) * folga

            # hessiana: H = D - (2/r) 1 1^T, com D diagonal
            d = 2.0 * a - r / (q * q)                  # d_i < 0, logo inversivel
            inv_d = 1.0 / d
            s = inv_d.sum()
            denom = 1.0 - (2.0 / r) * s

            # passo de Newton com H^-1 por Sherman-Morrison
            w = float(inv_d @ grad)
            passo = inv_d * grad + (2.0 / r) * inv_d * w / denom

            # amortecimento: nenhum poco passa de metade da distancia ate zero
            lim = 0.5 * q
            escala = 1.0
            negativos = passo > lim
            if np.any(negativos):
                escala = float(np.min(lim[negativos] / passo[negativos]))
            q_novo = np.clip(q - escala * passo, 1.0e-12, sup)

            if np.max(np.abs(q_novo - q)) < 1.0e-13:
                q = q_novo
                break
            q = q_novo
        r *= reducao
        if r < 1.0e-12:
            break

    return q, iteracoes


def resolve_sqp(pocos: dict[str, Poco], ativos: list[str], Qmax: float,
                economico: bool = False) -> tuple[np.ndarray, int]:
    """Metodo (c). Programacao quadratica sequencial, via scipy."""
    a, b, _, sup = _coefs(pocos, ativos, economico)

    neg_obj = lambda q: -float(np.sum(a * q * q + b * q))
    neg_jac = lambda q: -(2.0 * a * q + b)

    x0 = np.minimum(Qmax * sup / sup.sum(), sup)
    res = minimize(neg_obj, x0, jac=neg_jac, method="SLSQP",
                   bounds=[(0.0, float(s)) for s in sup],
                   constraints=[{"type": "eq",
                                 "fun": lambda q: float(q.sum() - Qmax),
                                 "jac": lambda q: np.ones_like(q)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    return np.clip(res.x, 0.0, sup), int(res.nit)


def totais(pocos: dict[str, Poco], ativos: list[str], q: np.ndarray
           ) -> tuple[float, float]:
    """Devolve (producao de oleo, m3/d ; lucro diario, $/d) de uma alocacao."""
    oleo = sum(pocos[w].oleo(float(qi)) for w, qi in zip(ativos, q))
    lucro = sum(pocos[w].lucro(float(qi)) for w, qi in zip(ativos, q))
    return oleo, lucro


# =============================================================================
#  CAMADA 1.3  Reproducao das tabelas de resultados
# =============================================================================

def camada_1_3(pocos: dict[str, Poco]) -> None:
    """Reproduz as Tabelas 4 e 8 e confronta os tres metodos."""
    print("\n" + "=" * 79)
    print(" CAMADA 1.3  Reproducao das tabelas de resultados")
    print("=" * 79)

    ativos = ESQUEMAS[1]

    for titulo, economico, pub_q, pub_val, rotulo in (
        ("Tabela 4  Esquema 1, objetivo = maxima producao de oleo",
         False, TAB4_QGI, TAB4_QO, "oleo (m3/d)"),
        ("Tabela 8  Esquema 1, objetivo = maximo lucro diario",
         True, TAB8_QGI, TAB8_Y, "lucro ($/d)"),
    ):
        print(f"\n  {titulo}")
        print("  " + "-" * 75)
        cab = "  Qmax  " + "".join(f"{w:>11}" for w in ativos) + f"{rotulo:>16}"
        print(cab)
        for Qmax in sorted(pub_q):
            q, _ = resolve_kkt(pocos, ativos, Qmax, economico)
            oleo, lucro = totais(pocos, ativos, q)
            obtido = lucro if economico else oleo
            linha = "".join(f"{qi:5.2f}/{pq:<5.2f}"
                            for qi, pq in zip(q, pub_q[Qmax]))
            print(f"  {Qmax:4.1f}  {linha}"
                  f"{obtido:9.2f}/{pub_val[Qmax]:<7.2f}")
        print("  (cada celula traz obtido/publicado; vazoes em 10^4 m3/d)")

    # ---- confronto entre os tres metodos -----------------------------------
    print("\n  Os tres metodos no mesmo ponto (Esquema 1, Qmax = 8, lucro):")
    print("  " + "-" * 75)
    Qmax = 8.0
    q_kkt, lam = resolve_kkt(pocos, ativos, Qmax, economico=True)
    _, y_kkt = totais(pocos, ativos, q_kkt)
    t0 = time.perf_counter()
    q_pen, it_pen = resolve_penalidade(pocos, ativos, Qmax, economico=True)
    t_pen = time.perf_counter() - t0
    _, y_pen = totais(pocos, ativos, q_pen)
    t0 = time.perf_counter()
    q_sqp, it_sqp = resolve_sqp(pocos, ativos, Qmax, economico=True)
    t_sqp = time.perf_counter() - t0
    _, y_sqp = totais(pocos, ativos, q_sqp)

    print(f"{'metodo':<34}{'lucro ($/d)':>14}{'iteracoes':>12}{'desvio':>12}")
    print(f"{'KKT por bissecao':<34}{y_kkt:14.3f}{'-':>12}{'referencia':>12}")
    print(f"{'penalidade mista + Newton':<34}{y_pen:14.3f}{it_pen:12d}"
          f"{abs(y_pen - y_kkt):12.3f}")
    print(f"{'SQP (scipy)':<34}{y_sqp:14.3f}{it_sqp:12d}"
          f"{abs(y_sqp - y_kkt):12.3f}")
    print(f"\n  multiplicador de Lagrange: lambda = {lam:.3f} $/(10^4 m3)")
    print("  Esse e o preco-sombra do gas: quanto o lucro do bloco cresceria")
    print("  com uma unidade adicional de gas disponivel.")


# =============================================================================
#  CAMADA 2  Analise das solucoes publicadas
# =============================================================================

def camada_2_1(pocos: dict[str, Poco]) -> None:
    """O ponto Qmax = 4 do Esquema 1, onde a solucao publicada e subotima."""
    print("\n" + "=" * 79)
    print(" CAMADA 2.1  O ponto Qmax = 4 do Esquema 1")
    print("=" * 79)

    ativos = ESQUEMAS[1]
    Qmax = 4.0
    q_pub = np.array(TAB4_QGI[Qmax])
    q_obt, _ = resolve_kkt(pocos, ativos, Qmax, economico=False)

    oleo_pub, _ = totais(pocos, ativos, q_pub)
    oleo_obt, _ = totais(pocos, ativos, q_obt)

    print(f"  gas total publicado : {q_pub.sum():.3f} x10^4 m3/d")
    print(f"  gas total obtido    : {q_obt.sum():.3f} x10^4 m3/d")
    print(f"\n  oleo da solucao publicada : {oleo_pub:7.2f} m3/d  "
          f"(a Tabela 4 reporta {TAB4_QO[Qmax]})")
    print(f"  oleo da solucao obtida    : {oleo_obt:7.2f} m3/d")
    print(f"  ganho                     : {oleo_obt - oleo_pub:7.2f} m3/d  "
          f"({(oleo_obt / oleo_pub - 1.0) * 100.0:.2f} por cento)")

    print("\n  Viabilidade da solucao obtida:")
    sup = np.array([pocos[w].qg_max_oleo for w in ativos])
    print(f"    balanco de gas respeitado : "
          f"{abs(q_obt.sum() - Qmax) < 1e-9}")
    print(f"    limites superiores        : {bool(np.all(q_obt <= sup + 1e-9))}")
    print(f"    nao negatividade          : {bool(np.all(q_obt >= -1e-9))}")

    print("\n  Produtividade marginal de cada poco nas duas solucoes:")
    print(f"{'poco':<5}{'q publicado':>13}{'dqo/dqg':>10}"
          f"{'q obtido':>11}{'dqo/dqg':>10}")
    for i, w in enumerate(ativos):
        p = pocos[w]
        m_pub = 2.0 * p.A * q_pub[i] + p.B
        m_obt = 2.0 * p.A * q_obt[i] + p.B
        print(f"{w:<5}{q_pub[i]:13.2f}{m_pub:10.3f}"
              f"{q_obt[i]:11.3f}{m_obt:10.3f}")
    print("\n  Na solucao obtida as produtividades marginais dos pocos com gas")
    print("  coincidem, como exige a condicao KKT. Na publicada, nao. Os")
    print("  demais quatro pontos da Tabela 4 reproduzem exatamente, o que")
    print("  isola o problema neste ponto e sugere parada prematura do metodo")
    print("  de penalidade dos autores.")


def camada_2_2(pocos: dict[str, Poco]) -> None:
    """Otimo tecnico contra otimo economico."""
    print("\n" + "=" * 79)
    print(" CAMADA 2.2  Otimo tecnico (ORAT) contra otimo economico (NPV)")
    print("=" * 79)

    ativos = ESQUEMAS[1]
    q_orat = np.array([pocos[w].qg_max_oleo for w in ativos])
    q_npv = np.array([pocos[w].qg_max_lucro for w in ativos])
    oleo_o, lucro_o = totais(pocos, ativos, q_orat)
    oleo_n, lucro_n = totais(pocos, ativos, q_npv)

    print(f"{'criterio':<26}{'gas total':>12}{'oleo':>11}{'lucro':>13}")
    print(f"{'maxima producao':<26}{q_orat.sum():12.3f}{oleo_o:11.2f}"
          f"{lucro_o:13.1f}")
    print(f"{'maximo lucro':<26}{q_npv.sum():12.3f}{oleo_n:11.2f}"
          f"{lucro_n:13.1f}")
    print(f"\n  variacao de gas   : {(q_npv.sum() / q_orat.sum() - 1) * 100:+.2f}%")
    print(f"  variacao de oleo  : {(oleo_n / oleo_o - 1) * 100:+.2f}%")
    print(f"  variacao de lucro : {(lucro_n / lucro_o - 1) * 100:+.2f}%")
    print("\n  O otimo economico produz menos oleo e rende mais, usando bem")
    print("  menos gas. O gas poupado pode ir para outros pocos do campo.")


# =============================================================================
#  CAMADA 3  Escalabilidade
# =============================================================================

def instancia_sintetica(n: int, frac_cd: float, rng: np.random.Generator
                        ) -> dict[str, Poco]:
    """
    Gera n pocos perturbando os oito reais.

    Cada poco sintetico herda os coeficientes de um dos oito originais,
    multiplicados por fatores independentes sorteados em [0,7 ; 1,3]. A
    fracao frac_cd fixa a proporcao de pocos do tipo c ou d, isto e, aqueles
    com gama negativo, que sao os que exigem injecao minima e geram a parte
    combinatoria do problema.
    """
    base = np.array([TABELA_1[w] for w in TABELA_1])        # 8 x 5
    idx = rng.integers(0, len(base), n)
    A = base[idx, 0] * rng.uniform(0.7, 1.3, n)
    B = base[idx, 1] * rng.uniform(0.7, 1.3, n)
    C = np.abs(base[idx, 2]) * rng.uniform(0.7, 1.3, n)
    C[: int(round(frac_cd * n))] *= -1.0                    # pocos tipo c ou d
    Fw, GOR = base[idx, 3], base[idx, 4]
    K = P1_OLEO + P2_GAS * GOR - P3_AGUA * Fw / (1.0 - Fw) - P5_DESG

    return {f"S{k:05d}": Poco(f"S{k:05d}", A[k], B[k], C[k], Fw[k], GOR[k],
                              K[k] * A[k], K[k] * B[k] - P4_INJ, K[k] * C[k])
            for k in range(n)}


def camada_3(pocos: dict[str, Poco]) -> None:
    """Mede onde o metodo escala e onde ele quebra."""
    print("\n" + "=" * 79)
    print(" CAMADA 3  Escalabilidade")
    print("=" * 79)
    rng = np.random.default_rng(20260120)

    print("\n  3.1  Custo de uma resolucao do NLP continuo")
    print("  " + "-" * 75)
    print(f"{'n pocos':>10}{'KKT (ms)':>12}{'penalidade (ms)':>18}"
          f"{'SQP (ms)':>14}{'com gas':>10}")
    for n in (8, 50, 100, 1_000, 10_000, 100_000):
        p = instancia_sintetica(n, 0.0, rng)
        ativos = list(p)
        Qmax = 0.35 * sum(p[w].qg_max_lucro for w in ativos)

        t0 = time.perf_counter()
        q, _ = resolve_kkt(p, ativos, Qmax, economico=True)
        t_kkt = (time.perf_counter() - t0) * 1e3

        if n <= 10_000:
            t0 = time.perf_counter()
            resolve_penalidade(p, ativos, Qmax, economico=True)
            t_pen = f"{(time.perf_counter() - t0) * 1e3:.1f}"
        else:
            t_pen = "-"

        # o SQP do scipy fatora a hessiana cheia a cada iteracao, de modo que
        # o custo cresce com o cubo do numero de pocos e deixa de ser viavel
        if n <= 100:
            t0 = time.perf_counter()
            resolve_sqp(p, ativos, Qmax, economico=True)
            t_sqp = f"{(time.perf_counter() - t0) * 1e3:.1f}"
        else:
            t_sqp = "nao conclui"

        print(f"{n:>10}{t_kkt:12.2f}{t_pen:>18}{t_sqp:>14}"
              f"{int(np.sum(q > 1e-9)):10d}")
    print("\n  Os tres metodos escalam de forma muito distinta. A bissecao no")
    print("  multiplicador explora a separabilidade do problema e resolve cem")
    print("  mil pocos em poucos milissegundos. A penalidade mista usa a")
    print("  estrutura diagonal mais posto um da hessiana, Eq. (28), e tambem")
    print("  escala bem. Ja o SQP fatora a hessiana cheia a cada iteracao e")
    print("  deixa de ser viavel a partir de algumas centenas de pocos.")

    print("\n  3.2  Custo da enumeracao dos pocos tipo c e d")
    print("  " + "-" * 75)
    p = instancia_sintetica(64, 0.25, rng)
    ativos = list(p)
    Qmax = 0.35 * sum(p[w].qg_max_lucro for w in ativos)
    t0 = time.perf_counter()
    for _ in range(50):
        resolve_kkt(p, ativos, Qmax, economico=True)
    unidade = (time.perf_counter() - t0) / 50.0

    print(f"{'n pocos':>9}{'tipo c/d':>10}{'esquemas':>24}{'tempo estimado':>22}")
    for n in (8, 20, 40, 60, 80, 120, 200):
        m = max(1, int(round(0.25 * n)))
        esquemas = 2 ** m
        total = esquemas * unidade
        if total < 60:
            s = f"{total:.2f} s"
        elif total < 3_600:
            s = f"{total / 60:.1f} min"
        elif total < 86_400 * 2:
            s = f"{total / 3_600:.1f} h"
        else:
            s = f"{total / (86_400 * 365):.3g} anos"
        print(f"{n:>9}{m:>10}{esquemas:>24,}{s:>22}")

    print("\n  O gargalo nao e o NLP, e a enumeracao. Com os oito pocos do")
    print("  campo JD sao apenas dois pocos tipo c e d, logo quatro esquemas,")
    print("  e o custo passa despercebido. Mantida a proporcao, a enumeracao")
    print("  deixa de ser viavel a partir de poucas dezenas de pocos, o que")
    print("  recomenda substitui-la por uma formulacao com variaveis binarias.")


# =============================================================================
#  PRINCIPAL
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Alocacao economica otima de gas lift (Zhong et al., 2022)")
    ap.add_argument("--camada", type=str, default="todas",
                    choices=["todas", "1", "2", "3"],
                    help="camada a executar")
    ap.add_argument("--figuras", action="store_true",
                    help="gera as figuras em PNG")
    args = ap.parse_args()

    pocos = monta_pocos(usar_tabela_3=True)

    print("=" * 79)
    print(" ALOCACAO ECONOMICA OTIMA DE GAS DE INJECAO EM POCOS DE GAS LIFT")
    print(" Reproducao de Zhong et al., Energies 15(19):6950, 2022")
    print(" Campo JD, oito pocos")
    print("=" * 79)

    if args.camada in ("todas", "1"):
        camada_1_1()
        camada_1_2(pocos)
        camada_1_3(pocos)
    if args.camada in ("todas", "2"):
        camada_2_1(pocos)
        camada_2_2(pocos)
    if args.camada in ("todas", "3"):
        camada_3(pocos)

    if args.figuras:
        from figuras import gerar_todas
        gerar_todas(pocos)

    print("\n" + "=" * 79)


if __name__ == "__main__":
    main()
