import numpy as np
from scipy.optimize import brentq

T1 = {'W1':(-0.58,3.86,1.082),'W2':(-0.92,4.24,3.530),'W3':(-1.42,6.53,2.091),
      'W4':(-1.35,4.68,6.889),'W5':(-0.287,2.36,2.893),'W6':(-1.106,5.68,0.788),
      'W7':(-1.62,7.36,-1.945),'W8':(-2.048,9.74,-3.022)}
W = list(T1)

def aloca(Qmax, coef, pocos):
    """KKT: 2a q + b = lam  ->  q = (lam-b)/2a, projetado em [0, -b/2a]."""
    def q_de(lam):
        out = {}
        for w in pocos:
            a, b = coef[w][0], coef[w][1]
            out[w] = min(max((lam-b)/(2*a), 0.0), -b/(2*a))
        return out
    f = lambda lam: sum(q_de(lam).values()) - Qmax
    lam = brentq(f, 0.0, max(c[1] for c in coef.values()), xtol=1e-12)
    return q_de(lam), lam

# ---- ORAT (Scheme 1, todos os 8 pocos), Tabela 4 do artigo
pub_q = {2.0:[0,0,0.35,0,0,0.06,0.56,1.03], 4.0:[0,0.11,0.88,0.24,0,0.74,1.03,1.00],
         8.0:[0.88,0.76,1.30,0.68,0,1.29,1.40,1.69], 12.0:[1.74,1.30,1.65,1.05,0.90,1.73,1.70,1.93],
         16.0:[2.45,1.75,1.94,1.35,2.33,2.11,1.96,2.13]}
pub_qo= {2.0:26.23, 4.0:35.61, 8.0:49.82, 12.0:58.97, 16.0:64.71}
print("== Reproducao da Tabela 4 (Scheme 1, objetivo = maxima producao de oleo) ==")
print(f"{'Qmax':>5} | " + " ".join(f"{w:>11}" for w in W) + f" | {'qo tot':>14}")
for Q in [2.0,4.0,8.0,12.0,16.0]:
    q,lam = aloca(Q, T1, W)
    qo = sum(T1[w][0]*q[w]**2 + T1[w][1]*q[w] + T1[w][2] for w in W)
    linha = " ".join(f"{q[w]:5.2f}/{pub_q[Q][i]:<5.2f}" for i,w in enumerate(W))
    print(f"{Q:5.1f} | {linha} | {qo:6.2f}/{pub_qo[Q]:<6.2f}")
    assert abs(sum(q.values())-Q) < 1e-9
print("\n(cada celula: obtido / publicado; vazao em 10^4 m3/d, oleo em m3/d)")

erros = []
for Q in [2.0,4.0,8.0,12.0,16.0]:
    q,_ = aloca(Q, T1, W)
    erros += [abs(q[w]-pub_q[Q][i]) for i,w in enumerate(W)]
    qo = sum(T1[w][0]*q[w]**2 + T1[w][1]*q[w] + T1[w][2] for w in W)
    erros.append(abs(qo-pub_qo[Q])/pub_qo[Q]*100)
print(f"\ndesvio maximo nas vazoes: {max(erros[:-1]):.3f} x10^4 m3/d (publicado com 2 casas)")
