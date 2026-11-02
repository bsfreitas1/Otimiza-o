#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refino da malha temporal do metodo de multiperiodos.

Mantem fixos o horizonte fisico (15 anos), a taxa de desconto anual (10 por
cento) e a calibracao das reservas. Varia apenas o passo de discretizacao, de
1 ano a 1 dia. O cronograma de fechamento e fixado em "nenhum poco fecha", que
e o otimo no caso anual, porque enumerar (T+1)^2 cronogramas seria inviavel nas
malhas finas.

Uso:  python malha.py <passo_em_dias> [<passo_em_dias> ...]
"""
from __future__ import annotations
import sys, time, json, math, os
sys.path.insert(0, '/home/claude/gaslift')
import numpy as np
from gaslift import (monta_pocos, POCOS, TABELA_1, P4_INJ, fator_economico,
                     resolve_kkt, Poco)
import deplecao as dp

HORIZONTE_DIAS = 15 * 365.0
TAXA = 0.10
QMAX = 8.0
CACHE = os.path.join(os.path.dirname(__file__), 'malha_resultados.json')

pocos = monta_pocos(True)
reservas = dp.calibra_reservas(pocos)
A = np.array([pocos[w].A for w in POCOS])
B = np.array([pocos[w].B for w in POCOS])
C = np.array([pocos[w].C for w in POCOS])
K = np.array([fator_economico(pocos[w].Fw, pocos[w].GOR) for w in POCOS])
Nmax = np.array([reservas[w] for w in POCOS])
sup = np.array([pocos[w].qg_max_oleo for w in POCOS])
qmin = np.array([
    (-B[k] + math.sqrt(B[k] ** 2 - 4.0 * A[k] * C[k])) / (2.0 * A[k])
    if C[k] < 0 else 0.0 for k in range(8)])


def campo_multiperiodos(passo_dias: float):
    """
    Metodo de campo multiperiodos. A cada periodo relevanta-se a curva com a
    depletacao acumulada e resolve-se o problema estatico, sem olhar o futuro.
    """
    T = int(round(HORIZONTE_DIAS / passo_dias))
    t0 = time.perf_counter()
    N = np.zeros(8)
    vpl = 0.0
    for t in range(T):
        d = np.clip(1.0 - N / Nmax, 0.0, None)
        alfa, beta, gama = K * A * d, K * B * d - P4_INJ, K * C * d
        pocos_t = {n: Poco(n, A[k] * d[k], B[k] * d[k], C[k] * d[k],
                           pocos[n].Fw, pocos[n].GOR, alfa[k], beta[k], gama[k])
                   for k, n in enumerate(POCOS)}
        q = np.zeros(8)
        viaveis = [n for k, n in enumerate(POCOS) if beta[k] > 0.0]
        melhor, melhor_val = None, -np.inf
        for liga7 in (True, False):
            for liga8 in (True, False):
                cand = [n for n in viaveis
                        if (n != "W7" or liga7) and (n != "W8" or liga8)]
                if not cand:
                    continue
                qc, _ = resolve_kkt(pocos_t, cand, QMAX, economico=True)
                val = sum(pocos_t[n].lucro(float(qi)) for n, qi in zip(cand, qc))
                if val > melhor_val:
                    melhor_val, melhor = val, (cand, qc)
        if melhor is not None:
            for n, qi in zip(*melhor):
                q[POCOS.index(n)] = float(qi)
        qo = np.maximum(d * (A * q ** 2 + B * q + C), 0.0)
        desc = 1.0 / (1.0 + TAXA) ** (t * passo_dias / 365.0)
        vpl += desc * passo_dias * float((K * qo - P4_INJ * q).sum())
        N = N + passo_dias * qo
    return vpl, time.perf_counter() - t0, T


def campo_estado_fino(passo_estado: float, passo_decisao: float = 365.0):
    """
    Metodo de campo multiperiodos com decisao anual e estado integrado em
    malha fina. A injecao e escolhida uma vez por ano, com a curva daquele
    momento, e mantida pelos 365 dias. O reservatorio, porem, deplete
    continuamente durante o ano, integrado em passos de 'passo_estado' dias.
    """
    t0 = time.perf_counter()
    n_dec = int(round(HORIZONTE_DIAS / passo_decisao))
    sub = max(1, int(round(passo_decisao / passo_estado)))
    h = passo_decisao / sub
    N = np.zeros(8)
    vpl = 0.0
    for t in range(n_dec):
        d = np.clip(1.0 - N / Nmax, 0.0, None)
        alfa, beta, gama = K * A * d, K * B * d - P4_INJ, K * C * d
        pocos_t = {n: Poco(n, A[k] * d[k], B[k] * d[k], C[k] * d[k],
                           pocos[n].Fw, pocos[n].GOR, alfa[k], beta[k], gama[k])
                   for k, n in enumerate(POCOS)}
        q = np.zeros(8)
        viaveis = [n for k, n in enumerate(POCOS) if beta[k] > 0.0]
        melhor, melhor_val = None, -np.inf
        for liga7 in (True, False):
            for liga8 in (True, False):
                cand = [n for n in viaveis
                        if (n != "W7" or liga7) and (n != "W8" or liga8)]
                if not cand:
                    continue
                qc, _ = resolve_kkt(pocos_t, cand, QMAX, economico=True)
                val = sum(pocos_t[n].lucro(float(qi)) for n, qi in zip(cand, qc))
                if val > melhor_val:
                    melhor_val, melhor = val, (cand, qc)
        if melhor is not None:
            for n, qi in zip(*melhor):
                q[POCOS.index(n)] = float(qi)
        # a injecao fica fixa, mas o reservatorio deplete durante o periodo
        for s in range(sub):
            dd = np.clip(1.0 - N / Nmax, 0.0, None)
            qo = np.maximum(dd * (A * q ** 2 + B * q + C), 0.0)
            dias = t * passo_decisao + s * h
            desc = 1.0 / (1.0 + TAXA) ** (dias / 365.0)
            vpl += desc * h * float((K * qo - P4_INJ * q).sum())
            N = N + h * qo
    return vpl, time.perf_counter() - t0, n_dec, sub


def multiperiodos_decisao_anual(passo_estado: float = 1.0,
                                passo_decisao: float = 365.0,
                                max_iter: int = 3000):
    """
    Versao B. A decisao e anual, como no metodo de campo, mas tomada com visao
    do horizonte inteiro. O estado e integrado em malha fina pelo mesmo Euler
    que o metodo de campo usa, de modo que a unica diferenca entre os dois
    passa a ser a antecipacao.
    """
    import casadi as ca
    n_dec = int(round(HORIZONTE_DIAS / passo_decisao))
    sub = max(1, int(round(passo_decisao / passo_estado)))
    h = passo_decisao / sub
    ESC = 1.0e6
    t0 = time.perf_counter()

    opti = ca.Opti()
    qg = opti.variable(8, n_dec)                 # so 15 decisoes por poco
    x = opti.variable(8, n_dec * sub + 1)        # estado em malha fina
    obj = 0
    opti.subject_to(x[:, 0] == 0)
    for t in range(n_dec):
        q = qg[:, t]
        curva = A * q ** 2 + B * q + C
        opti.subject_to(ca.sum1(q) <= QMAX)
        for s in range(sub):
            k = t * sub + s
            d = 1 - x[:, k]
            qo = d * curva
            opti.subject_to(qo >= 0)
            opti.subject_to(x[:, k + 1] == x[:, k] + h * qo / Nmax)
            dias = t * passo_decisao + s * h
            desc = 1.0 / (1.0 + TAXA) ** (dias / 365.0)
            obj += desc * h * ca.sum1(K * qo - P4_INJ * q) / ESC
    for k in range(8):
        if qmin[k] > 0:
            opti.subject_to(qg[k, :] >= qmin[k])
    opti.subject_to(opti.bounded(0, ca.vec(qg), np.tile(sup, n_dec)))
    opti.subject_to(opti.bounded(0, ca.vec(x), np.ones(8 * (n_dec * sub + 1))))
    opti.minimize(-obj)

    q0, _ = resolve_kkt(pocos, POCOS, QMAX, economico=True)
    ini = np.maximum(np.tile(np.asarray(q0).reshape(-1, 1), (1, n_dec)),
                     qmin.reshape(-1, 1))
    opti.set_initial(qg, ini)
    opti.set_initial(x, np.zeros((8, n_dec * sub + 1)))
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                          "ipopt.sb": "yes", "ipopt.max_iter": max_iter,
                          "ipopt.tol": 1e-9, "ipopt.acceptable_tol": 1e-7})
    sol = opti.solve()
    dt = time.perf_counter() - t0
    nvar = 8 * n_dec + 8 * (n_dec * sub + 1)
    return (float(sol.value(obj)) * ESC, dt, n_dec, sub, nvar,
            np.array(sol.value(qg)).reshape(8, n_dec))


def multiperiodos(passo_dias: float, max_iter: int = 3000):
    """
    Metodo de multiperiodos. Resolve o horizonte inteiro de uma vez, com a
    injecao constante dentro de cada periodo de 'passo_dias'.
    """
    import casadi as ca
    T = int(round(HORIZONTE_DIAS / passo_dias))
    ESC = 1.0e6
    t0 = time.perf_counter()
    opti = ca.Opti()
    qg = opti.variable(8, T)
    x = opti.variable(8, T + 1)          # estado adimensional, N / Nmax
    obj = 0
    opti.subject_to(x[:, 0] == 0)
    for t in range(T):
        d = 1 - x[:, t]
        curva = A * qg[:, t] ** 2 + B * qg[:, t] + C
        qo = d * curva
        opti.subject_to(qo >= 0)
        opti.subject_to(x[:, t + 1] == x[:, t] + passo_dias * qo / Nmax)
        opti.subject_to(ca.sum1(qg[:, t]) <= QMAX)
        desc = 1.0 / (1.0 + TAXA) ** (t * passo_dias / 365.0)
        obj += desc * passo_dias * ca.sum1(K * qo - P4_INJ * qg[:, t]) / ESC
    for k in range(8):
        if qmin[k] > 0:
            opti.subject_to(qg[k, :] >= qmin[k])
    opti.subject_to(opti.bounded(0, ca.vec(qg), np.tile(sup, T)))
    opti.subject_to(opti.bounded(0, ca.vec(x), np.ones(8 * (T + 1))))
    opti.minimize(-obj)

    q0, _ = resolve_kkt(pocos, POCOS, QMAX, economico=True)
    ini = np.maximum(np.tile(np.asarray(q0).reshape(-1, 1), (1, T)),
                     qmin.reshape(-1, 1))
    opti.set_initial(qg, ini)
    opti.set_initial(x, np.zeros((8, T + 1)))
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                          "ipopt.sb": "yes", "ipopt.max_iter": max_iter,
                          "ipopt.tol": 1e-9, "ipopt.acceptable_tol": 1e-7})
    sol = opti.solve()
    dt = time.perf_counter() - t0
    nvar = 8 * T + 8 * (T + 1)
    return float(sol.value(obj)) * ESC, dt, T, nvar


if __name__ == "__main__":
    dados = {}
    if os.path.exists(CACHE):
        dados = json.load(open(CACHE))
    for arg in sys.argv[1:]:
        passo = float(arg)
        if arg == "campo":
            continue
        vpl, dt, T, nvar = multiperiodos(passo)
        dados[arg] = {"passo_dias": passo, "T": T, "nvar": nvar,
                      "vpl": vpl, "tempo": dt}
        print(f"passo={passo:g} d  T={T}  vars={nvar}  "
              f"VPL={vpl/1e6:.6f} MUSD  tempo={dt:.2f} s")
        json.dump(dados, open(CACHE, "w"), indent=1)
