import numpy as np, casadi as ca

# Tabela 1: coef. GLPC (qo = A qg^2 + B qg + C), water cut, GOR
T1 = {  # poco: (A, B, C, Fw, GORf)
 'W1': (-0.58 , 3.86, 1.082, 0.52 , 0.016 ),
 'W2': (-0.92 , 4.24, 3.530, 0.45 , 0.010 ),
 'W3': (-1.42 , 6.53, 2.091, 0.48 , 0.0125),
 'W4': (-1.35 , 4.68, 6.889, 0.39 , 0.015 ),
 'W5': (-0.287, 2.36, 2.893, 0.58 , 0.018 ),
 'W6': (-1.106, 5.68, 0.788, 0.50 , 0.016 ),
 'W7': (-1.62 , 7.36,-1.945, 0.443, 0.020 ),
 'W8': (-2.048, 9.74,-3.022, 0.55 , 0.0175),
}
# Tabela 2: parametros economicos
p1, p2, p3, p4, p5 = 301.89, 2012.38, 0.077, 309.60, 0.0077
# Tabela 3: coef. GLEPC publicados (alfa, beta, gama)
T3 = {'W1':(-193.81, 980.26, 361.56),'W2':(-296.31,1055.96,1136.89),
      'W3':(-464.50,1826.40, 683.98),'W4':(-448.36,1244.70,2287.94),
      'W5':( -96.49, 488.58, 978.44),'W6':(-362.41,1588.40, 263.31),
      'W7':(-540.86,2208.20,-665.55),'W8':(-671.19,2975.99,-1018.99)}

print("== NIVEL 1: Tab.1 + Tab.2 --(Eqs. 6-8)--> Tab.3 ==")
print(f"{'poco':5} {'alfa calc':>10} {'alfa pub':>9} {'beta calc':>10} {'beta pub':>9} {'gama calc':>10} {'gama pub':>9}")
calc = {}
for w,(A,B,C,Fw,G) in T1.items():
    K = p1 + p2*G - p3*Fw/(1-Fw) - p5
    a, b, g = K*A, K*B - p4, K*C
    calc[w] = (a,b,g)
    pa,pb,pg = T3[w]
    print(f"{w:5} {a:10.2f} {pa:9.2f} {b:10.2f} {pb:9.2f} {g:10.2f} {pg:9.2f}")
err = max(max(abs(calc[w][k]-T3[w][k])/abs(T3[w][k]) for k in range(3)) for w in T1)
print(f"erro relativo maximo: {err*100:.3f}%\n")

print("== NIVEL 2: formula fechada vs. coluna 'Unconstrained' ==")
# ORAT (max oleo): qg* = -B/2A   -> Tabela 4
orat_pub = {'W1':3.33,'W2':2.30,'W3':2.30,'W4':1.73,'W5':4.11,'W6':2.57,'W7':2.27,'W8':2.38}
# NPV  (max lucro): qg* = -beta/2alfa -> Tabela 8
npv_pub  = {'W1':2.44,'W2':1.72,'W3':1.92,'W4':1.35,'W5':2.31,'W6':2.11,'W7':1.97,'W8':2.14}
print(f"{'poco':5} {'ORAT calc':>9} {'ORAT pub':>8} | {'NPV calc':>8} {'NPV pub':>7} {'dif %':>6}")
for w,(A,B,C,Fw,G) in T1.items():
    qo_star = -B/(2*A)
    a,b,_ = T3[w]
    qn_star = -b/(2*a)
    d = 100*(qn_star-npv_pub[w])/npv_pub[w]
    print(f"{w:5} {qo_star:9.3f} {orat_pub[w]:8.2f} | {qn_star:8.3f} {npv_pub[w]:7.2f} {d:+6.1f}")
