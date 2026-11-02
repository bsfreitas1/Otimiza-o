import numpy as np, time
from scipy.optimize import brentq

rng = np.random.default_rng(42)
# estatisticas dos 8 pocos reais (Tabela 1 do artigo)
A0 = np.array([-0.58,-0.92,-1.42,-1.35,-0.287,-1.106,-1.62,-2.048])
B0 = np.array([3.86,4.24,6.53,4.68,2.36,5.68,7.36,9.74])
C0 = np.array([1.082,3.530,2.091,6.889,2.893,0.788,-1.945,-3.022])

def instancia(n):
    """amostra n pocos perturbando os 8 reais (mantem ordem de grandeza e sinais)"""
    idx = rng.integers(0,8,n)
    A = A0[idx]*rng.uniform(0.7,1.3,n); B = B0[idx]*rng.uniform(0.7,1.3,n)
    C = C0[idx]*rng.uniform(0.7,1.3,n)
    return A,B,C

def kkt(A,B,Qmax):
    """bissecao no multiplicador lambda; O(n log(1/tol))"""
    qmax = -B/(2*A)
    q = lambda lam: np.clip((lam-B)/(2*A), 0, qmax)
    lam = brentq(lambda l: q(l).sum()-Qmax, 0.0, B.max(), xtol=1e-12)
    return q(lam)

print("== custo do NLP continuo (KKT por bissecao) ==")
for n in [8, 100, 1000, 10000, 100000]:
    A,B,C = instancia(n); Q = 0.35*(-B/(2*A)).sum()
    t0=time.perf_counter(); q=kkt(A,B,Q); dt=time.perf_counter()-t0
    print(f"n={n:>6}  t={dt*1000:7.2f} ms   pocos com gas: {(q>1e-9).sum():>6}")

print("\n== custo da ENUMERACAO dos pocos tipo c/d (2^m esquemas) ==")
print("assumindo 25% dos pocos tipo c/d, como no artigo (2 de 8)")
for n in [8, 20, 40, 80, 200]:
    m = max(1, round(0.25*n)); esquemas = 2**m
    # cada esquema custa um NLP; mede o custo unitario em n pocos
    A,B,C = instancia(n); Q = 0.35*(-B/(2*A)).sum()
    t0=time.perf_counter()
    for _ in range(20): kkt(A,B,Q)
    unit=(time.perf_counter()-t0)/20
    tot = esquemas*unit
    if   tot < 60:      s=f"{tot:.2f} s"
    elif tot < 3600:    s=f"{tot/60:.1f} min"
    elif tot < 86400*2: s=f"{tot/3600:.1f} h"
    else:               s=f"{tot/(86400*365):.3g} anos"
    print(f"n={n:>4}  m={m:>3} tipo c/d  esquemas=2^{m}={esquemas:>22,}  tempo estimado: {s}")
