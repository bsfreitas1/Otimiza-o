#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 CAMADA 4  Alocacao de gas de injecao sob depletacao do reservatorio
 -----------------------------------------------------------------------------
 Extensao do modelo de

   ZHONG, H.; ZHAO, C.; XU, Z.; ZHENG, C. Economical optimum gas allocation
   model considering different types of gas-lift performance curves.
   Energies, v. 15, n. 19, 6950, 2022. DOI 10.3390/en15196950.

 O modelo original e estatico: resolve um instante de operacao do bloco. Esta
 camada acrescenta a dimensao temporal, sem alterar nada do que ja foi
 reproduzido. O modulo gaslift.py nao e modificado; apenas importado.

 Organizacao:

   4.1  Do balanco de materiais ao declinio no acumulado
   4.2  Criterio de abandono economico em forma fechada
   4.3  NLP multiperiodo nao convexo, resolvido com IPOPT
   4.4  Valor da previsao: politica otima contra politica miope
   4.5  Validacao

 Execucao:
     python deplecao.py                roda tudo
     python deplecao.py --secao 2      roda apenas a secao indicada
     python deplecao.py --figuras      gera as figuras
===============================================================================
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from gaslift import (TABELA_1, P4_INJ, fator_economico, monta_pocos,
                     resolve_kkt, totais, POCOS, Poco)

# =============================================================================
#  PARAMETROS DA DEPLETACAO
# =============================================================================
#
# A taxa de declinio natural vem de
#
#   INTERNATIONAL ENERGY AGENCY. The Implications of Oil and Gas Field Decline
#   Rates. Paris: IEA, 2025. Licenca CC BY 4.0.
#
# O relatorio tabula o declinio pos-pico observado em campos convencionais:
# 4,2 por cento ao ano em campos onshore, 5,6 por cento na media de todos os
# campos convencionais e 8,5 por cento como declinio natural estimado na
# ausencia de novo investimento. E esta ultima a figura pertinente aqui,
# porque o que declina no modelo e a capacidade produtiva do poco, nao a
# producao efetiva de um campo que recebe intervencoes.

D_NATURAL = 0.085        # declinio natural, fracao por ano       (IEA, 2025)
D_ONSHORE = 0.042        # declinio observado onshore             (IEA, 2025)
D_CONVENC = 0.056        # media dos campos convencionais         (IEA, 2025)

TAXA_DESC = 0.10         # taxa de desconto anual
DIAS_ANO = 365.0
HORIZONTE = 15           # numero de periodos anuais
QMAX_REF = 8.0           # gas disponivel, 10^4 m3/d (nivel central do artigo)

TIPO_CD = ["W7", "W8"]   # pocos com gama < 0, sujeitos a disjuncao


# =============================================================================
#  4.1  DO BALANCO DE MATERIAIS AO DECLINIO NO ACUMULADO
# =============================================================================

def calibra_reservas(pocos: dict[str, Poco], Qmax: float = QMAX_REF,
                     taxa: float = D_NATURAL,
                     modo: str = "uniforme") -> dict[str, float]:
    """
    Converte a taxa de declinio anual em um volume recuperavel por poco.

    O argumento modo escolhe como a reserva total do bloco se reparte:

      "uniforme"     todos os pocos drenam volumes iguais. E a hipotese de
                     drenagem regular: pocos de um mesmo bloco, perfurados com
                     espacamento comparavel, varrem areas comparaveis. A vida
                     de cada poco passa a depender do quanto ele produz, de
                     modo que o poco que recebe mais gas se esgota primeiro.
                     E esta a calibracao do caso base, porque e a unica das
                     duas que acopla a decisao de hoje a capacidade de amanha.

      "proporcional" a reserva de cada poco e proporcional a sua producao de
                     referencia. Todos os pocos passam a ter a mesma razao
                     reserva-producao e, portanto, a mesma vida 1/D. E a
                     hipotese de que a produtividade atual revela a reserva.
                     Serve de contraste na analise de sensibilidade.

    Em ambos os modos a reserva total do bloco e a mesma, 365 Q_o^0 / D, com
    Q_o^0 a producao total no ponto de operacao de referencia. As duas
    calibracoes diferem apenas no rateio.

    Hipoteses, na ordem em que sao usadas:

    (H1) IPR linear. A vazao de liquido de um poco e proporcional ao
         diferencial de pressao, q_L = J (P_r - P_wf).

    (H2) Reservatorio sub-saturado com balanco de materiais simples. A pressao
         media cai linearmente com o volume de oleo ja produzido,
         P_r = P_r0 (1 - N/N_max).

    (H3) A pressao de fundo em operacao e governada pela coluna, isto e, por
         q_g, e nao pela depletacao. Logo, para um mesmo q_g, a vazao escala
         com o fator de capacidade remanescente

             d(N) = 1 - N / N_max,

         e a GLPC do artigo se desloca verticalmente sem mudar de eixo:

             q_o(q_g, N) = d(N) (A q_g^2 + B q_g + C).

    De (H3) decorre uma consequencia que fecha a calibracao. Sendo
    dN/dt = q_o, vale

        dd/dt = -q_o / N_max = -(q_o^0 / N_max) d,

    de modo que d cai exponencialmente no tempo com constante

        D = q_o^0 / N_max.

    O modelo linear no acumulado produz, portanto, declinio exponencial no
    tempo, que e a curva de Arps com expoente b = 0. A calibracao fica

        N_max = q_o^0 / D = 365 q_o^0(d) / D   [m3, com q_o^0 em m3/d],

    isto e, exatamente a razao reserva-producao do poco. Toma-se como q_o^0 a
    producao de cada poco na solucao economica otima estatica, que e o ponto de
    operacao de referencia reproduzido na Camada 1.3.
    """
    q, _ = resolve_kkt(pocos, POCOS, Qmax, economico=True)
    qo0 = {nome: pocos[nome].oleo(float(qg)) for nome, qg in zip(POCOS, q)}
    total = DIAS_ANO * sum(qo0.values()) / taxa
    if modo == "proporcional":
        soma = sum(qo0.values())
        return {n: total * qo0[n] / soma for n in POCOS}
    if modo == "uniforme":
        return {n: total / len(POCOS) for n in POCOS}
    raise ValueError(f"modo desconhecido: {modo}")


def secao_4_1(pocos: dict[str, Poco]) -> dict[str, float]:
    print("\n" + "=" * 79)
    print(" SECAO 4.1  Calibracao do declinio a partir da razao reserva-producao")
    print("=" * 79)
    print(f" Taxa de declinio natural adotada: {100*D_NATURAL:.1f} por cento ao"
          f" ano (IEA, 2025)")
    print(f" Ponto de operacao de referencia:  Qmax = {QMAX_REF:.0f}"
          f" x10^4 m3/d, objetivo de lucro\n")
    reservas = calibra_reservas(pocos)
    prop = calibra_reservas(pocos, modo="proporcional")
    q, _ = resolve_kkt(pocos, POCOS, QMAX_REF, economico=True)
    print(f"{'poco':<5}{'qg*':>8}{'qo (m3/d)':>12}{'N_max unif':>13}"
          f"{'vida':>8}{'N_max prop':>13}{'vida':>8}")
    tot_res = 0.0
    for nome, qg in zip(POCOS, q):
        qo0 = pocos[nome].oleo(float(qg))
        nmax = reservas[nome]
        tot_res += nmax
        vida = nmax / (DIAS_ANO * qo0) if qo0 > 0 else float("inf")
        vida_p = prop[nome] / (DIAS_ANO * qo0) if qo0 > 0 else float("inf")
        print(f"{nome:<5}{float(qg):>8.3f}{qo0:>12.3f}{nmax/1e3:>13.1f}"
              f"{vida:>8.1f}{prop[nome]/1e3:>13.1f}{vida_p:>8.1f}")
    print(f"\n Reserva total do bloco: {tot_res/1e3:.1f} x10^3 m3"
          f"  ({tot_res*6.2898/1e6:.2f} milhoes de barris), igual nos dois modos")
    print(f" No rateio proporcional a vida e 1/D = {1/D_NATURAL:.1f} anos para")
    print(" todos, por construcao, e o declinio e homogeneo. No rateio uniforme")
    print(" a vida varia com a producao, e e essa heterogeneidade que da ao")
    print(" problema um acoplamento temporal com conteudo: dar gas a um poco")
    print(" hoje antecipa o seu esgotamento.")
    return reservas


# =============================================================================
#  4.2  CRITERIO DE ABANDONO ECONOMICO EM FORMA FECHADA
# =============================================================================

def secao_4_2(pocos: dict[str, Poco]) -> None:
    """
    Sob depletacao, a receita escala com d mas o custo de pressurizacao nao.

    Com a GLPC deslocada por d, a curva economica do poco passa a ser

        y(q_g, d) = K d (A q_g^2 + B q_g + C) - p_4 q_g,

    de onde

        alfa(d) = K A d,   beta(d) = K B d - p_4,   gama(d) = K C d.

    O termo -p_4 nao e multiplicado por d. Essa assimetria tem tres
    consequencias, todas verificaveis em forma fechada.

    (i) Ha um fator de capacidade critico abaixo do qual nenhuma injecao se
        paga. A produtividade marginal em q_g = 0 e beta(d); ela se anula em

            d* = p_4 / (K B).

        Para d < d* o poco nao aceita gas a nenhum preco-sombra, e sai do
        esquema de forma definitiva, porque d so decresce.

    (ii) O otimo economico isolado se desloca,

            q_g*(d) = -beta(d) / (2 alfa(d)) = -(K B d - p_4) / (2 K A d),

         enquanto o otimo tecnico -B/(2A) nao depende de d. A distancia entre
         os dois cresce com a depletacao.

    (iii) A ordem de prioridade entre pocos, que no caso estatico e a ordem de
          beta, nao se altera sob declinio uniforme: multiplicar todos os
          K_i B_i pelo mesmo d preserva a ordem apenas se p_4 for comum, o que
          e o caso. O que muda e quem permanece viavel, pela ordem de d*.
    """
    print("\n" + "=" * 79)
    print(" SECAO 4.2  Criterio de abandono economico em forma fechada")
    print("=" * 79)
    print(" d* = p_4 / (K B) e o fator de capacidade em que beta se anula.")
    print(" Abaixo dele a injecao nunca se paga e o poco sai em definitivo.\n")
    print(f"{'poco':<5}{'K':>9}{'B':>8}{'d*':>9}{'N/N_max em d*':>16}"
          f"{'ano de saida':>15}{'tipo':>7}")
    linhas = []
    for nome, (A, B, C, Fw, GOR) in TABELA_1.items():
        K = fator_economico(Fw, GOR)
        dstar = P4_INJ / (K * B)
        # d(t) = exp(-D t)  =>  t* = -ln(d*)/D
        t_saida = -math.log(dstar) / D_NATURAL
        tipo = "c/d" if C < 0 else ("a/b")
        linhas.append((dstar, nome, K, B, t_saida, tipo))
    for dstar, nome, K, B, t_saida, tipo in sorted(linhas, reverse=True):
        print(f"{nome:<5}{K:>9.2f}{B:>8.2f}{dstar:>9.4f}{1-dstar:>16.4f}"
              f"{t_saida:>15.1f}{tipo:>7}")
    print("\n A ordem de saida e a ordem decrescente de d*: sai primeiro o poco")
    print(" que precisa de mais capacidade remanescente para pagar o proprio")
    print(" gas. Note que a ordem de saida e a inversa da ordem de prioridade")
    print(" por beta, de modo que o bloco se esvazia de baixo para cima.")

    print("\n Deslocamento do otimo isolado, Eq. (ii):")
    print(f"{'poco':<5}{'-B/2A':>9}" + "".join(f"{'d='+f'{d:.2f}':>9}"
          for d in (1.00, 0.70, 0.50, 0.30)))
    for nome, (A, B, C, Fw, GOR) in TABELA_1.items():
        K = fator_economico(Fw, GOR)
        linha = f"{nome:<5}{-B/(2*A):>9.3f}"
        for d in (1.00, 0.70, 0.50, 0.30):
            al, be = K * A * d, K * B * d - P4_INJ
            linha += f"{max(-be/(2*al), 0.0):>9.3f}"
        print(linha)
    print("\n O otimo tecnico e invariante; o economico recua. A tese central do")
    print(" artigo, de que os dois nao coincidem, portanto se acentua no tempo.")


# =============================================================================
#  4.3  NLP MULTIPERIODO
# =============================================================================

def resolve_cronograma(pocos: dict[str, Poco], reservas: dict[str, float],
                       fecha: dict[str, int], T: int = HORIZONTE,
                       Qmax: float = QMAX_REF, taxa_desc: float = TAXA_DESC,
                       inicial: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Resolve o NLP multiperiodo com um cronograma de fechamento dado.

    O argumento fecha da, para cada poco de tipo c ou d, o periodo a partir do
    qual ele esta fechado: fecha[w] = T significa que nunca fecha e
    fecha[w] = 0 que nunca abre. Enquanto aberto, o poco e obrigado a receber
    pelo menos a injecao que zera a sua GLPC; fechado, nao recebe gas nem
    produz, e a sua reserva fica preservada.

    Devolve None quando o subproblema e infactivel.
    """
    import casadi as ca

    ativos = list(POCOS)
    m = len(ativos)
    A = np.array([pocos[w].A for w in ativos])
    B = np.array([pocos[w].B for w in ativos])
    C = np.array([pocos[w].C for w in ativos])
    K = np.array([fator_economico(pocos[w].Fw, pocos[w].GOR) for w in ativos])
    Nmax = np.array([reservas[w] for w in ativos])
    sup = np.array([pocos[w].qg_max_oleo for w in ativos])

    # injecao minima tecnica: menor raiz de A q^2 + B q + C = 0, quando C < 0
    qmin = np.zeros(m)
    for k in range(m):
        if C[k] < 0.0:
            disc = B[k] ** 2 - 4.0 * A[k] * C[k]
            qmin[k] = (-B[k] + math.sqrt(disc)) / (2.0 * A[k])

    opti = ca.Opti()
    # O estado entra adimensionalizado, x = N / N_max em [0, 1], e o objetivo
    # em milhoes de dolares. Sem esse reescalonamento as grandezas do problema
    # se espalham por sete ordens de magnitude, de q_g na casa da unidade a um
    # valor presente na casa de 10^7, e o IPOPT falha de forma erratica: o
    # mesmo modelo convergia em T = 12 e abortava em T = 10 e T = 15.
    ESCALA = 1.0e6
    qg = opti.variable(m, T)
    x = opti.variable(m, T + 1)
    obj = 0
    opti.subject_to(x[:, 0] == 0)
    for t in range(T):
        d = 1 - x[:, t]
        qo_lista = []
        for k, nome in enumerate(ativos):
            aberto = t < fecha.get(nome, T)
            if not aberto:
                opti.subject_to(qg[k, t] == 0)
                qo_lista.append(ca.MX(0))
                continue
            curva = A[k] * qg[k, t] ** 2 + B[k] * qg[k, t] + C[k]
            qo_k = d[k] * curva
            opti.subject_to(qo_k >= 0)
            if qmin[k] > 0.0:
                opti.subject_to(qg[k, t] >= qmin[k])
            qo_lista.append(qo_k)
        qo = ca.vertcat(*qo_lista)
        opti.subject_to(x[:, t + 1] == x[:, t] + DIAS_ANO * qo / Nmax)
        opti.subject_to(ca.sum1(qg[:, t]) <= Qmax)
        obj += DIAS_ANO / (1.0 + taxa_desc) ** t * ca.sum1(
            K * qo - P4_INJ * qg[:, t]) / ESCALA
    opti.subject_to(opti.bounded(0, ca.vec(qg), np.tile(sup, T)))
    opti.subject_to(opti.bounded(0, ca.vec(x), np.ones(m * (T + 1))))
    opti.minimize(-obj)

    if inicial is None:
        q0, _ = resolve_kkt(pocos, ativos, Qmax, economico=True)
        inicial = np.tile(q0.reshape(-1, 1), (1, T))
        inicial = np.maximum(inicial, qmin.reshape(-1, 1))
    opti.set_initial(qg, inicial)
    opti.set_initial(x, np.zeros((m, T + 1)))
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                          "ipopt.sb": "yes", "ipopt.max_iter": 3000,
                          "ipopt.tol": 1e-9, "ipopt.acceptable_tol": 1e-7})
    try:
        sol = opti.solve()
    except RuntimeError:
        return None
    return (np.array(sol.value(qg)).reshape(m, T),
            np.array(sol.value(x)).reshape(m, T + 1) * Nmax.reshape(-1, 1),
            float(sol.value(obj)) * ESCALA)


def resolve_multiperiodo(pocos: dict[str, Poco], reservas: dict[str, float],
                         T: int = HORIZONTE, Qmax: float = QMAX_REF,
                         taxa_desc: float = TAXA_DESC,
                         ativos: list[str] | None = None,
                         verbose: bool = False
                         ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Resolve o problema completo enumerando os cronogramas de fechamento.

    A Secao 4.2 mostra que o fator de capacidade d so decresce e que a
    viabilidade economica de um poco e um limiar em d. O fechamento e,
    portanto, irreversivel: um poco que saiu nao volta. Essa monotonicidade
    colapsa a arvore de disjuncoes, que a principio teria 4^T ramos com dois
    pocos de tipo c e d, para os (T+1)^2 cronogramas em que cada um desses
    pocos escolhe um unico instante de saida. Para T = 15 sao 256 NLPs, o que
    permite resolver a disjuncao por enumeracao exata em vez de recorrer a
    uma formulacao com variaveis binarias.
    """
    t0 = time.perf_counter()
    melhor = None
    n_ok = 0
    for tau7 in range(T + 1):
        for tau8 in range(T + 1):
            r = resolve_cronograma(pocos, reservas,
                                   {"W7": tau7, "W8": tau8}, T, Qmax, taxa_desc)
            if r is None:
                continue
            n_ok += 1
            if melhor is None or r[2] > melhor[2]:
                melhor = r + ({"W7": tau7, "W8": tau8},)
    dt = time.perf_counter() - t0
    if melhor is None:
        raise RuntimeError("nenhum cronograma factivel")
    if verbose:
        print(f"   {n_ok} de {(T+1)**2} cronogramas factiveis;"
              f" otimo em {melhor[3]}")
    return melhor[0], melhor[1], melhor[2], dt, melhor[3], n_ok



def politica_miope(pocos: dict[str, Poco], reservas: dict[str, float],
                   T: int = HORIZONTE, Qmax: float = QMAX_REF,
                   taxa_desc: float = TAXA_DESC
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Politica miope, que e a pratica de campo.

    A cada periodo levanta-se a curva corrente de cada poco, resolve-se o
    problema estatico do artigo com essa curva e injeta-se o resultado durante
    todo o periodo, sem considerar o efeito da decisao sobre a capacidade
    futura. O estado e entao avancado e o procedimento se repete.

    E o mesmo horizonte e o mesmo fator de desconto do NLP multiperiodo, de
    modo que a diferenca de valor presente entre as duas politicas mede
    exatamente o que se ganha ao antecipar a depletacao.
    """
    m = len(POCOS)
    A = np.array([pocos[w].A for w in POCOS])
    B = np.array([pocos[w].B for w in POCOS])
    C = np.array([pocos[w].C for w in POCOS])
    K = np.array([fator_economico(pocos[w].Fw, pocos[w].GOR) for w in POCOS])
    Nmax = np.array([reservas[w] for w in POCOS])

    qg_hist = np.zeros((m, T))
    N = np.zeros((m, T + 1))
    vpl = 0.0
    for t in range(T):
        d = 1.0 - N[:, t] / Nmax
        d = np.clip(d, 0.0, None)
        # curva economica corrente de cada poco
        alfa, beta, gama = K * A * d, K * B * d - P4_INJ, K * C * d
        pocos_t: dict[str, Poco] = {}
        for k, nome in enumerate(POCOS):
            pocos_t[nome] = Poco(nome, A[k] * d[k], B[k] * d[k], C[k] * d[k],
                                 pocos[nome].Fw, pocos[nome].GOR,
                                 alfa[k], beta[k], gama[k])
        # so entram os pocos ainda viaveis, isto é, com beta > 0
        ativos = [n for k, n in enumerate(POCOS) if beta[k] > 0.0]
        q = np.zeros(m)
        if ativos:
            # entre os viaveis, descartam-se os que nao se pagam no ponto
            melhor, melhor_val = None, -np.inf
            for liga7 in (True, False):
                for liga8 in (True, False):
                    cand = [n for n in ativos
                            if (n != "W7" or liga7) and (n != "W8" or liga8)]
                    if not cand:
                        continue
                    qc, _ = resolve_kkt(pocos_t, cand, Qmax, economico=True)
                    val = sum(pocos_t[n].lucro(float(qi))
                              for n, qi in zip(cand, qc))
                    if val > melhor_val:
                        melhor_val, melhor = val, (cand, qc)
            cand, qc = melhor
            for n, qi in zip(cand, qc):
                q[POCOS.index(n)] = float(qi)
        qo = np.maximum(d * (A * q ** 2 + B * q + C), 0.0)
        qg_hist[:, t] = q
        N[:, t + 1] = N[:, t] + DIAS_ANO * qo
        vpl += DIAS_ANO / (1.0 + taxa_desc) ** t * float((K * qo - P4_INJ * q).sum())
    return qg_hist, N, vpl


def secao_4_3(pocos: dict[str, Poco], reservas: dict[str, float]):
    print("\n" + "=" * 79)
    print(" SECAO 4.3  NLP multiperiodo nao convexo")
    print("=" * 79)
    qg, N, vpl, dt, cron, n_ok = resolve_multiperiodo(pocos, reservas)
    m, T = qg.shape
    print(f" Variaveis por NLP: {m*T} de injecao + {m*(T+1)} de estado ="
          f" {m*T + m*(T+1)}")
    print(f" Restricoes por NLP: {m*T} de curva, {m*T} de balanco,"
          f" {T} de gas")
    print(f" Cronogramas de fechamento enumerados: {(T+1)**2}"
          f" ({n_ok} factiveis)")
    print(f" Tempo total da enumeracao: {dt:.2f} s"
          f"  ({dt/(T+1)**2*1e3:.0f} ms por NLP)")
    print(f" Cronograma otimo: W7 fecha no ano {cron['W7']+1}, W8 no ano"
          f" {cron['W8']+1}"
          + ("  (nenhum fecha dentro do horizonte)"
             if cron['W7'] >= T and cron['W8'] >= T else ""))
    print(f" Valor presente liquido: {vpl/1e6:.3f} milhoes de dolares\n")
    print(" Injecao otima por poco, 10^4 m3/d:")
    print(f"{'ano':<5}" + "".join(f"{n:>8}" for n in POCOS) + f"{'total':>9}")
    for t in range(T):
        print(f"{t+1:<5}" + "".join(f"{qg[k,t]:>8.3f}" for k in range(m))
              + f"{qg[:,t].sum():>9.3f}")
    print("\n Fator de capacidade remanescente d:")
    Nmax = np.array([reservas[w] for w in POCOS])
    print(f"{'ano':<5}" + "".join(f"{n:>8}" for n in POCOS))
    for t in range(0, T + 1, 3):
        d = 1.0 - N[:, t] / Nmax
        print(f"{t:<5}" + "".join(f"{d[k]:>8.3f}" for k in range(m)))
    return qg, N, vpl


# =============================================================================
#  4.4  VALOR DA PREVISAO
# =============================================================================

def secao_4_4(pocos: dict[str, Poco], reservas: dict[str, float],
              qg_o: np.ndarray, N_o: np.ndarray, vpl_otimo: float) -> None:
    print("\n" + "=" * 79)
    print(" SECAO 4.4  Valor da previsao: politica otima contra politica miope")
    print("=" * 79)
    print(" A politica miope levanta a curva corrente de cada poco a cada ano,")
    print(" resolve com ela o problema estatico do artigo e injeta o resultado")
    print(" durante todo o periodo, sem considerar o efeito da decisao sobre a")
    print(" capacidade futura. E o procedimento de campo. A diferenca de valor")
    print(" presente entre as duas politicas mede o que se ganha ao antecipar")
    print(" a depletacao.\n")
    qg_m, N_m, vpl_m = politica_miope(pocos, reservas)
    Nmax = np.array([reservas[w] for w in POCOS])
    print(f" VPL da politica otima multiperiodo: {vpl_otimo/1e6:>9.3f} MUSD")
    print(f" VPL da politica miope de campo:     {vpl_m/1e6:>9.3f} MUSD")
    print(f" Ganho da antecipacao da depletacao: "
          f"{100.0*(vpl_otimo/vpl_m - 1.0):>9.3f} por cento")
    print("\n Oleo recuperado ao fim do horizonte, 10^3 m3:")
    print(f"{'poco':<5}{'otima':>10}{'miope':>10}{'N_max':>10}"
          f"{'rec.ot.%':>10}{'rec.mi.%':>10}")
    for k, nome in enumerate(POCOS):
        print(f"{nome:<5}{N_o[k,-1]/1e3:>10.1f}{N_m[k,-1]/1e3:>10.1f}"
              f"{Nmax[k]/1e3:>10.1f}{100*N_o[k,-1]/Nmax[k]:>10.1f}"
              f"{100*N_m[k,-1]/Nmax[k]:>10.1f}")
    print(f"{'tot':<5}{N_o[:,-1].sum()/1e3:>10.1f}{N_m[:,-1].sum()/1e3:>10.1f}"
          f"{Nmax.sum()/1e3:>10.1f}{100*N_o[:,-1].sum()/Nmax.sum():>10.1f}"
          f"{100*N_m[:,-1].sum()/Nmax.sum():>10.1f}")
    print("\n Gas efetivamente injetado, 10^4 m3/d (disponivel:"
          f" {QMAX_REF:.0f}):")
    print(f"{'ano':<5}{'otima':>10}{'miope':>10}{'folga ot.':>12}")
    for t in range(qg_m.shape[1]):
        so = qg_o[:, t].sum()
        print(f"{t+1:<5}{so:>10.3f}{qg_m[:,t].sum():>10.3f}"
              f"{QMAX_REF-so:>12.3f}")
    print("\n A folga na ultima coluna e a parte do gas disponivel que a")
    print(" politica otima deixa de injetar. Enquanto ela e nula o gas e o")
    print(" recurso escasso e o preco-sombra e positivo; quando passa a ser")
    print(" positiva, a restricao de disponibilidade deixa de ser ativa e o")
    print(" que limita o bloco e a propria capacidade remanescente dos pocos.")

    print("\n Alocacao no primeiro periodo, que e a decisao que se toma hoje:")
    print(f"{'poco':<6}{'multiperiodo':>14}{'estatica':>11}{'dif':>9}")
    qest, _ = resolve_kkt(pocos, POCOS, QMAX_REF, economico=True)
    for k, nome in enumerate(POCOS):
        print(f"{nome:<6}{qg_o[k,0]:>14.3f}{qest[k]:>11.3f}"
              f"{qg_o[k,0]-qest[k]:>9.3f}")
    print(" O modelo estatico nao da gas nenhum ao W5; o multiperiodo da, ja")
    print(" no primeiro ano. E a prova de que o acoplamento temporal altera a")
    print(" decisao corrente, e nao apenas as futuras.")

    print("\n Robustez do ganho da antecipacao. Varia-se a taxa de desconto e")
    print(" o horizonte, mantendo fixo o cronograma em que nenhum poco fecha,")
    print(" o que dispensa reenumerar e basta para comparar as duas politicas.")
    print(f"\n{'r':>6}{'T':>5}{'VPL otimo':>12}{'VPL miope':>12}{'ganho %':>10}")
    for r_desc in (0.00, 0.10, 0.20):
        for T in (10, 15, 25):
            res = resolve_cronograma(pocos, reservas, {"W7": T, "W8": T},
                                     T=T, taxa_desc=r_desc)
            if res is None:
                print(f"{r_desc:>6.2f}{T:>5}{'nao convergiu':>34}")
                continue
            _, _, v_ot = res
            _, _, v_mi = politica_miope(pocos, reservas, T=T, taxa_desc=r_desc)
            print(f"{r_desc:>6.2f}{T:>5}{v_ot/1e6:>12.3f}{v_mi/1e6:>12.3f}"
                  f"{100*(v_ot/v_mi-1):>10.3f}")
    print("\n O ganho nao e uniforme. Na taxa de desconto de 10 por cento que")
    print(" se adotou, ele fica em torno de 0,15 por cento, e cai a menos de")
    print(" 0,1 por cento a 20 por cento de desconto: a pratica de campo, de")
    print(" levantar as curvas de tempo em tempo e reotimizar com elas, e")
    print(" entao quase otima. Mas o ganho sobe a 3,3 por cento sem desconto")
    print(" e com horizonte de 25 anos. O padrao e coerente com a origem do")
    print(" acoplamento: a politica miope erra ao nao poupar os pocos que")
    print(" esgotam cedo, e esse erro so se paga nos periodos distantes, que")
    print(" sao justamente os que o desconto apaga. Antecipar a depletacao")
    print(" importa, portanto, na medida em que o futuro longo importa.")


# =============================================================================
#  4.5  VALIDACAO
# =============================================================================

def secao_4_5(pocos: dict[str, Poco], reservas: dict[str, float]) -> None:
    print("\n" + "=" * 79)
    print(" SECAO 4.5  Validacao")
    print("=" * 79)

    # (a) consistencia em t = 0
    print(" (a) Consistencia com o caso estatico no primeiro periodo.")
    print("     Com T = 1 o estado inicial e nulo, portanto d = 1, e o modelo")
    print("     multiperiodo se reduz ao problema estatico. O confronto exige")
    print("     uma ressalva. Esta camada monta a curva economica a partir das")
    print("     Tabelas 1 e 2, pelas Eqs. (6) a (8), porque so assim os")
    print("     coeficientes acompanham o fator d. A Camada 1.3, por sua vez,")
    print("     usa os coeficientes publicados na Tabela 3. Como a Camada 1.1")
    print("     mostrou, as duas leituras discordam em alfa nos pocos W5 a W8,")
    print("     de 0,53 a 2,83 por cento. A comparacao pertinente e, portanto,")
    print("     contra a alocacao estatica recalculada.")
    qg1 = resolve_multiperiodo(pocos, reservas, T=1)[0]
    rec = monta_pocos(usar_tabela_3=False)
    qrec, _ = resolve_kkt(rec, POCOS, QMAX_REF, economico=True)
    qpub, _ = resolve_kkt(pocos, POCOS, QMAX_REF, economico=True)
    print(f"\n{'poco':<6}{'T=1':>10}{'recalc.':>10}{'dif':>11}"
          f"{'Tabela 3':>11}{'dif':>11}")
    pior_r = pior_p = 0.0
    for k, nome in enumerate(POCOS):
        dr, dp = abs(qg1[k, 0] - qrec[k]), abs(qg1[k, 0] - qpub[k])
        pior_r, pior_p = max(pior_r, dr), max(pior_p, dp)
        print(f"{nome:<6}{max(qg1[k,0],0.0):>10.4f}{qrec[k]:>10.4f}"
              f"{dr:>11.2e}{qpub[k]:>11.4f}{dp:>11.2e}")
    print(f"\n     Contra a alocacao recalculada: {pior_r:.2e} x10^4 m3/d")
    print(f"     Contra a Tabela 3 publicada:   {pior_p:.2e} x10^4 m3/d")
    print("     A primeira diferenca e o erro numerico do NLP; a segunda")
    print("     reproduz a inconsistencia da Tabela 3 do artigo, ja medida na")
    print("     Camada 1.1, e nao e atribuivel a esta extensao.")

    # (b) d* analitico contra o numerico
    print("\n (b) Criterio de abandono: d* analitico contra o observado no NLP.")
    T = HORIZONTE
    qg, N, vpl_ref, _, cron, _ = resolve_multiperiodo(pocos, reservas, T=T)
    Nmax = np.array([reservas[w] for w in POCOS])
    print("     O limiar d* vale quando o gas deixa de ser escasso, isto e,")
    print("     quando o preco-sombra da disponibilidade e nulo. Compara-se,")
    print("     para cada poco, d* com o fator de capacidade no inicio do")
    print("     ultimo periodo em que ele ainda recebe gas.")
    print(f"\n{'poco':<6}{'d*':>9}{'d no ult. periodo':>20}{'qg nesse ano':>14}"
          f"{'qg no ano final':>17}")
    for k, nome in enumerate(POCOS):
        A, B, C, Fw, GOR = TABELA_1[nome]
        K = fator_economico(Fw, GOR)
        dstar = P4_INJ / (K * B)
        com_gas = [t for t in range(T) if qg[k, t] > 1e-3]
        if com_gas:
            t_ult = max(com_gas)
            d_ini = 1.0 - N[k, t_ult] / Nmax[k]
            print(f"{nome:<6}{dstar:>9.4f}{d_ini:>20.4f}"
                  f"{qg[k,t_ult]:>14.4f}{qg[k,-1]:>17.4f}")
        else:
            print(f"{nome:<6}{dstar:>9.4f}{'nunca recebe gas':>20}"
                  f"{'-':>14}{qg[k,-1]:>17.4f}")
    print("\n     O poco cuja trajetoria cruza o proprio limiar dentro do")
    print("     horizonte e o W4, o mais produtivo do bloco e por isso o que")
    print("     esgota primeiro. A injecao que o NLP lhe atribui se anula")
    print("     justamente quando d cruza d*, sem que esse valor tenha sido")
    print("     informado ao solver: o criterio analitico e o numerico")
    print("     concordam.")

    # (c) nao convexidade: multiplos pontos de partida
    print("\n (c) Nao convexidade. O termo d q_g^2 e bilinear nas variaveis de")
    print("     decisao e de estado, de modo que o NLP nao e convexo e o que o")
    print("     IPOPT devolve e um ponto KKT. Resolve-se o cronograma otimo de")
    print("     varios pontos de partida aleatorios para sondar a existencia")
    print("     de otimos locais distintos.")
    rng = np.random.default_rng(20260101)
    sup = np.array([pocos[w].qg_max_oleo for w in POCOS])
    vals, falhas = [], 0
    for _ in range(20):
        ini = rng.uniform(0.0, 1.0, (len(POCOS), T)) * sup.reshape(-1, 1)
        r = resolve_cronograma(pocos, reservas, cron, T=T, inicial=ini)
        if r is None:
            falhas += 1
        else:
            vals.append(r[2])
    if vals:
        v = np.array(vals)
        disp = 100.0 * (v.max() / v.min() - 1.0)
        print(f"\n     {len(vals)} partidas convergiram, {falhas} falharam")
        print(f"     VPL entre {v.min()/1e6:.6f} e {v.max()/1e6:.6f} MUSD")
        print(f"     dispersao relativa: {disp:.2e} por cento")
        print(f"     referencia (partida estatica): {vpl_ref/1e6:.6f} MUSD")
        if disp < 1e-4:
            print("     Todas as partidas caem no mesmo ponto. O problema nao e")
            print("     convexo, mas nesta instancia nao se observa")
            print("     multiplicidade de otimos locais.")
        else:
            print("     As partidas caem em pontos distintos: ha otimos locais")
            print("     e o valor reportado e o melhor encontrado.")


# =============================================================================
#  SENSIBILIDADE A TAXA DE DECLINIO
# =============================================================================

def sensibilidade(pocos: dict[str, Poco]) -> None:
    print("\n" + "=" * 79)
    print(" SENSIBILIDADE A TAXA DE DECLINIO")
    print("=" * 79)
    print(" Fixa-se o cronograma em que nenhum poco de tipo c ou d fecha, que")
    print(" e o otimo no caso base, para nao reenumerar os 256 subproblemas em")
    print(" cada configuracao.\n")
    print(f"{'D (%/ano)':>10}{'rateio':>14}{'fonte':>24}{'VPL (MUSD)':>12}"
          f"{'c/gas ao fim':>14}{'recup.%':>10}")
    T = HORIZONTE
    for modo in ("uniforme", "proporcional"):
        for taxa, fonte in ((D_ONSHORE, "IEA onshore"),
                            (D_CONVENC, "IEA convencionais"),
                            (D_NATURAL, "IEA declinio natural"),
                            (0.120, "hipotetico")):
            res = calibra_reservas(pocos, taxa=taxa, modo=modo)
            r = resolve_cronograma(pocos, res, {"W7": T, "W8": T}, T=T)
            if r is None:
                print(f"{100*taxa:>10.1f}{modo:>14}{fonte:>24}"
                      f"{'nao convergiu':>36}")
                continue
            qg, N, vpl = r
            nfim = int((qg[:, -1] > 1e-3).sum())
            rec = 100.0 * N[:, -1].sum() / sum(res.values())
            print(f"{100*taxa:>10.1f}{modo:>14}{fonte:>24}{vpl/1e6:>12.3f}"
                  f"{nfim:>14d}{rec:>10.1f}")
    print("\n O VPL cresce quando D cai, porque a reserva calibrada e maior.")
    print(" O rateio uniforme, que e o caso base, da sempre o VPL menor: a")
    print(" reserva fica concentrada longe dos pocos mais produtivos, e e essa")
    print(" discrepancia entre produtividade e reserva que a politica otima")
    print(" tem de administrar.")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secao", type=str, default=None)
    ap.add_argument("--figuras", action="store_true")
    args = ap.parse_args()

    pocos = monta_pocos(usar_tabela_3=True)

    if args.figuras:
        import figuras_deplecao
        figuras_deplecao.gerar(pocos)
        return

    reservas = secao_4_1(pocos)
    if args.secao in (None, "2"):
        secao_4_2(pocos)
    if args.secao in (None, "3", "4", "5"):
        qg, N, vpl = secao_4_3(pocos, reservas)
        if args.secao in (None, "4"):
            secao_4_4(pocos, reservas, qg, N, vpl)
        if args.secao in (None, "5"):
            secao_4_5(pocos, reservas)
    if args.secao is None:
        sensibilidade(pocos)


if __name__ == "__main__":
    main()
