"""
Blendagem de minerio de fosfato por programacao linear.

Reproducao do modelo de:
    ALI, M.A.M.; WASLY, H.S.; ABDELLAH, W.R.; JANG, H.
    Optimization of blending operation for Aswan phosphate mines using
    linear programming. Mining of Mineral Deposits, v. 12, n. 4, p. 1-8, 2018.
    DOI: 10.15407/mining12.04.001

Implementado com PuLP + CBC (os autores usaram o Solver do Excel).

NOTA SOBRE A FORMULACAO
-----------------------
A formulacao impressa no artigo apresenta duas inconsistencias:

  (a) As restricoes de teor (Eqs. 10-13 do artigo) somam apenas seis das
      oito variaveis no membro direito -- os indices 2 e 3 estao ausentes --
      e o texto da Secao 3.1 refere "twelve decision variables" quando a
      formulacao apresentada envolve trinta e duas.

  (b) Os percentuais da Tabela 8 do artigo estao normalizados POR PRODUTO
      (cada coluna soma 100%), e nao pela reserva total. Essa convencao
      nao e declarada pelos autores.

O modelo abaixo e uma reconstrucao coerente com os dados das Tabelas 6 e 7
e com o significado fisico do problema. A reconstrucao e validada pela
reproducao da Tabela 8 (ver funcao validar()).

Uso:
    python fosfato.py                 # resolve, valida e imprime o relatorio
    python fosfato.py --figura        # gera tambem a figura dos resultados
    python fosfato.py --sem-blendagem # inclui a politica de referencia
"""

import argparse
from dataclasses import dataclass, field

import pulp

# =====================================================================
# 1. DADOS DE ENTRADA
# =====================================================================

# --- Tabela 6 do artigo: frentes de lavra ---
# (operacao, camada, reserva Q_i [t], teor medio g_i [% P2O5])
FRENTES = [
    ("B1", "Terceira",   67400, 22.5),
    ("B1", "Superior",  215311, 23.6),
    ("B1", "Inferior",  192096, 23.6),
    ("B2", "Terceira",   19441, 21.0),
    ("B2", "Superior",  112374, 23.5),
    ("B2", "Inferior",  132127, 24.3),
    ("C",  "Superior",  801150, 24.0),
    ("C",  "Inferior", 1115627, 24.2),
]

NOMES = [f"{op}-{cam}" for op, cam, _, _ in FRENTES]
OPERACAO = {f"{op}-{cam}": op for op, cam, _, _ in FRENTES}
Q = {f"{op}-{cam}": q for op, cam, q, _ in FRENTES}      # reserva, t
G = {f"{op}-{cam}": g for op, cam, _, g in FRENTES}      # teor, % P2O5
RESERVA_TOTAL = sum(Q.values())
# Obs.: a soma das oito reservas da Tabela 6 e 2.655.526 t; o artigo reporta
# 2.655.527 t no total da mesma tabela (diferenca de 1 t, provavelmente
# arredondamento). A diferenca nao afeta os resultados.

# --- Tabela 7 do artigo: faixas comerciais ---
FAIXAS = [24, 23, 22, 21]                 # teor-alvo tau_j, % P2O5
PRECO = {24: 15.50, 23: 12.50, 22: 10.50, 21: 5.00}      # p_j, US$/t

# --- Tabela 8 do artigo: gabarito de validacao (% por produto) ---
# Apenas as celulas nao nulas; as demais sao zero.
TABELA8 = {
    ("B1-Terceira", 24):  1.12, ("B1-Terceira", 22): 66.10,
    ("B1-Superior", 24):  8.28, ("B1-Superior", 22):  0.33,
    ("B1-Inferior", 24):  7.40,
    ("B2-Terceira", 22): 33.57,
    ("B2-Superior", 24):  4.33,
    ("B2-Inferior", 24):  5.09,
    ("C-Superior",  24): 30.84,
    ("C-Inferior",  24): 42.95,
}

# Agregados por operacao para o produto de 24%, citados no resumo do artigo
AGREGADO_REF = {"B1": 16.80, "B2": 9.42, "C": 73.78}


# =====================================================================
# 2. MODELO
# =====================================================================

@dataclass
class Solucao:
    """Resultado de uma execucao do modelo."""
    rotulo: str
    status: str
    receita: float
    x: dict = field(repr=False, default_factory=dict)      # (frente, faixa) -> t
    n_variaveis: int = 0
    n_restricoes: int = 0
    tempo_s: float = 0.0

    def massa_por_faixa(self):
        """M_j = soma_i x_ij."""
        return {j: sum(self.x[(i, j)] for i in NOMES) for j in FAIXAS}

    def percentuais(self, tol=1e-6):
        """Converte para percentuais POR PRODUTO (convencao da Tabela 8)."""
        M = self.massa_por_faixa()
        return {(i, j): (100 * self.x[(i, j)] / M[j] if M[j] > tol else 0.0)
                for i in NOMES for j in FAIXAS}

    def teor_resultante(self, tol=1e-6):
        """Confere o teor medio efetivo de cada produto."""
        M = self.massa_por_faixa()
        return {j: (sum(G[i] * self.x[(i, j)] for i in NOMES) / M[j]
                    if M[j] > tol else None)
                for j in FAIXAS}


def resolver(rotulo="base", teor_exato=True, consumir_tudo=False,
             desempate=False, msg=False):
    """Monta e resolve o modelo de programacao linear.

    Parametros
    ----------
    teor_exato : bool
        True  -> Eq. (4): teor medio da mistura IGUAL ao alvo.
        False -> teor medio MAIOR OU IGUAL ao alvo (leitura alternativa).
    consumir_tudo : bool
        True  -> Eq. (2) como igualdade: toda a reserva e lavrada.
        False -> Eq. (2) como desigualdade (leitura adotada).
    desempate : bool
        O modelo possui solucoes otimas alternativas. Quando True, aplica-se
        um segundo estagio lexicografico que, entre as solucoes de receita
        maxima, destina o maximo possivel da frente de menor teor ao produto
        de menor valor -- criterio operacionalmente natural, uma vez que esse
        material nao tem valor comercial isolado. Esse e o vertice reportado
        em [1].

    As duas leituras alternativas servem a analise de robustez descrita
    na Secao 2.3 do resumo estendido.
    """
    prob = pulp.LpProblem("blendagem_" + rotulo.replace(" ", "_")
                           .replace(",", ""), pulp.LpMaximize)

    # --- Variaveis de decisao: x_ij >= 0, continuas (Eq. 5) ---
    x = pulp.LpVariable.dicts("x", (NOMES, FAIXAS), lowBound=0,
                              cat=pulp.LpContinuous)

    # --- Eq. (1): maximizar a receita total ---
    prob += pulp.lpSum(PRECO[j] * x[i][j] for i in NOMES for j in FAIXAS), \
        "receita_total"

    # --- Eq. (2): disponibilidade de reserva por frente ---
    for i in NOMES:
        soma = pulp.lpSum(x[i][j] for j in FAIXAS)
        if consumir_tudo:
            prob += soma == Q[i], f"reserva_{i}"
        else:
            prob += soma <= Q[i], f"reserva_{i}"

    # --- Eq. (4): enquadramento de teor ---
    # Forma linearizada da condicao de media ponderada (Eq. 3):
    #     sum_i g_i x_ij / sum_i x_ij = tau_j
    # multiplicada pelo denominador e reagrupada.
    for j in FAIXAS:
        desvio = pulp.lpSum((G[i] - j) * x[i][j] for i in NOMES)
        if teor_exato:
            prob += desvio == 0, f"teor_{j}"
        else:
            prob += desvio >= 0, f"teor_min_{j}"

    # --- Segundo estagio lexicografico (desempate) ---
    if desempate:
        prob.solve(pulp.PULP_CBC_CMD(msg=msg))
        if prob.objective.value() is not None:
            receita_otima = pulp.value(prob.objective)
            pior = min(NOMES, key=lambda i: G[i])      # frente de menor teor
            alvo_baixo = min(j for j in FAIXAS
                             if sum(x[i][j].value() for i in NOMES) > 1e-6)
            prob += (pulp.lpSum(PRECO[j] * x[i][j]
                                for i in NOMES for j in FAIXAS)
                     >= receita_otima - 1e-4), "mantem_receita"
            prob.setObjective(x[pior][alvo_baixo])
            prob.sense = pulp.LpMaximize

    # --- Resolucao pelo metodo Simplex ---
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))

    if desempate and prob.objective.value() is not None:
        receita = sum(PRECO[j] * x[i][j].value()
                      for i in NOMES for j in FAIXAS)
        return Solucao(
            rotulo=rotulo, status=pulp.LpStatus[prob.status], receita=receita,
            x={(i, j): x[i][j].value() for i in NOMES for j in FAIXAS},
            n_variaveis=len(prob.variables()),
            n_restricoes=len(prob.constraints),
            tempo_s=prob.solutionTime,
        )

    status = pulp.LpStatus[prob.status]
    if prob.objective.value() is None:
        return Solucao(rotulo, status, float("nan"))

    return Solucao(
        rotulo=rotulo,
        status=status,
        receita=pulp.value(prob.objective),
        x={(i, j): x[i][j].value() for i in NOMES for j in FAIXAS},
        n_variaveis=len(prob.variables()),
        n_restricoes=len(prob.constraints),
        tempo_s=prob.solutionTime,
    )


# =====================================================================
# 3. POLITICA DE REFERENCIA (contrafactual, fora do artigo)
# =====================================================================

def sem_blendagem():
    """Receita obtida sem misturar: cada frente e vendida pela faixa
    correspondente ao seu proprio teor, arredondado para baixo.

    Serve para quantificar em dolares o ganho atribuivel a blendagem.
    Frentes com teor abaixo da menor faixa comercial nao tem valor.
    """
    receita = 0.0
    detalhe = {}
    menor_faixa = min(FAIXAS)
    for i in NOMES:
        aplicaveis = [j for j in FAIXAS if j <= G[i]]
        if aplicaveis:
            faixa = max(aplicaveis)          # melhor faixa que o teor atinge
            valor = PRECO[faixa] * Q[i]
        else:
            faixa, valor = None, 0.0         # abaixo do minimo comercial
        detalhe[i] = (faixa, valor)
        receita += valor
    return receita, detalhe, menor_faixa


# =====================================================================
# 4. RELATORIO E VALIDACAO
# =====================================================================

def imprimir_dados():
    print("\nDADOS DE ENTRADA (Tabelas 6 e 7 do artigo)")
    print(f"{'Frente':<14}{'Reserva (t)':>13}{'Teor (%)':>10}")
    print("-" * 37)
    for i in NOMES:
        print(f"{i:<14}{Q[i]:>13,}{G[i]:>10.1f}".replace(",", "."))
    print("-" * 37)
    print(f"{'Total':<14}{RESERVA_TOTAL:>13,}".replace(",", "."))
    print(f"\nPrecos por faixa (US$/t): "
          + ", ".join(f"{j}% = {PRECO[j]:.2f}" for j in FAIXAS))


def imprimir_solucao(s):
    M = s.massa_por_faixa()
    teor = s.teor_resultante()
    ativas = [j for j in FAIXAS if M[j] > 1e-6]

    print(f"\nReceita otima: US$ {s.receita:,.0f}".replace(",", "."))
    print(f"Dimensao: {s.n_variaveis} variaveis, {s.n_restricoes} restricoes"
          f" | tempo: {s.tempo_s:.3f} s | status: {s.status}")

    print("\nMassa alocada por faixa comercial:")
    for j in FAIXAS:
        marca = "" if M[j] > 1e-6 else "   (nao utilizada)"
        print(f"  {j}%: {M[j]:>12,.0f} t"
              f"  ({100*M[j]/RESERVA_TOTAL:5.2f}% da reserva){marca}"
              .replace(",", "."))

    print("\nConferencia do teor resultante (deve igualar o alvo):")
    for j in ativas:
        print(f"  faixa {j}%: teor obtido = {teor[j]:.4f}%")

    pct = s.percentuais()
    print("\nPERCENTUAIS DE BLENDAGEM POR PRODUTO (%)")
    cab = "Frente".ljust(14) + "".join(f"{j}%".rjust(9) for j in ativas)
    print(cab)
    print("-" * len(cab))
    for i in NOMES:
        linha = f"{i:<14}"
        for j in ativas:
            v = pct[(i, j)]
            linha += f"{v:>9.2f}" if v > 1e-6 else f"{'-':>9}"
        print(linha)


def validar(s):
    """Compara a solucao obtida com os resultados publicados.

    A comparacao e feita em tres niveis, porque o problema apresenta
    solucoes otimas alternativas (degenerescencia): a receita, as massas
    por faixa e os agregados por operacao sao unicos, mas a composicao
    interna do produto de 22% admite mais de um vertice otimo.
    """
    pct = s.percentuais()
    M = s.massa_por_faixa()

    print("\nVALIDACAO CONTRA O ARTIGO")

    # --- Nivel 1: agregados por operacao no produto de 24% ---
    print("\n(a) Agregado por operacao no produto de 24%:")
    erro_ag = 0.0
    for op in ("B1", "B2", "C"):
        obt = 100 * sum(s.x[(i, 24)] for i in NOMES
                        if OPERACAO[i] == op) / M[24]
        ref = AGREGADO_REF[op]
        erro_ag += abs(obt - ref)
        print(f"    {op}: obtido {obt:6.2f}%  |  artigo {ref:6.2f}%")
    print(f"    erro acumulado: {erro_ag:.2f} p.p.")

    # --- Nivel 2: matriz completa de percentuais ---
    print("\n(b) Percentuais por frente e produto:")
    print(f"    {'Frente':<14}{'obt.24%':>9}{'ref.24%':>9}"
          f"{'obt.22%':>9}{'ref.22%':>9}")
    print("    " + "-" * 50)
    erro24 = erro22 = 0.0
    for i in NOMES:
        linha = f"    {i:<14}"
        for j in (24, 22):
            obtido = pct[(i, j)]
            ref = TABELA8.get((i, j), 0.0)
            if j == 24:
                erro24 += abs(obtido - ref)
            else:
                erro22 += abs(obtido - ref)
            linha += f"{obtido:>9.2f}{ref:>9.2f}"
        print(linha)
    print("    " + "-" * 50)
    print(f"    erro no produto de 24%: {erro24:.2f} p.p.")
    print(f"    erro no produto de 22%: {erro22:.2f} p.p.")

    # --- Nivel 3: leitura do resultado ---
    print("\n(c) Interpretacao:")
    if erro24 < 0.5:
        print("    O produto de 24%, que concentra "
              f"{100*M[24]/RESERVA_TOTAL:.1f}% da reserva, e reproduzido")
        print("    dentro do arredondamento das casas decimais publicadas.")
    if erro22 > 0.5:
        print("    O produto de 22% admite solucoes otimas alternativas: a")
        print("    massa total da faixa e unica, mas sua composicao entre as")
        print("    frentes de teor baixo nao e. Solucoes distintas produzem")
        print("    a mesma receita, e a faixa representa apenas "
              f"{100*M[22]/RESERVA_TOTAL:.2f}% da reserva.")
    return erro24 + erro22


def demonstrar_degenerescencia():
    """Evidencia a existencia de solucoes otimas alternativas.

    Fixa a receita no valor otimo e otimiza objetivos secundarios opostos
    sobre o conjunto de solucoes otimas. Se os resultados diferirem, ha
    mais de um vertice otimo.
    """
    base = resolver("base")
    print("\nDEMONSTRACAO DE SOLUCOES OTIMAS ALTERNATIVAS")
    print("Fixa-se a receita no otimo e otimiza-se um objetivo secundario.\n")

    resultados = {}
    for sentido, nome in ((pulp.LpMaximize, "max"), (pulp.LpMinimize, "min")):
        prob = pulp.LpProblem("degen_" + nome, sentido)
        x = pulp.LpVariable.dicts("x", (NOMES, FAIXAS), lowBound=0)
        # objetivo secundario: participacao da frente de menor teor no 22%
        prob += x["B2-Terceira"][22]
        prob += (pulp.lpSum(PRECO[j] * x[i][j] for i in NOMES for j in FAIXAS)
                 >= base.receita - 1e-4), "receita_otima"
        for i in NOMES:
            prob += pulp.lpSum(x[i][j] for j in FAIXAS) <= Q[i]
        for j in FAIXAS:
            prob += pulp.lpSum((G[i] - j) * x[i][j] for i in NOMES) == 0
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        M22 = sum(x[i][22].value() for i in NOMES)
        resultados[nome] = (
            pulp.value(prob.objective),
            100 * x["B2-Terceira"][22].value() / M22 if M22 > 1e-6 else 0.0,
        )

    for nome, (massa, pc) in resultados.items():
        print(f"  {nome} massa de B2-Terceira no produto 22%: "
              f"{massa:>10,.0f} t  ({pc:5.2f}% do produto)".replace(",", "."))
    faixa = abs(resultados["max"][1] - resultados["min"][1])
    if faixa > 1e-3:
        print(f"\n  A participacao varia em {faixa:.2f} p.p. sem alterar a")
        print("  receita: o modelo possui solucoes otimas alternativas.")
    else:
        print("\n  Nao foram detectadas solucoes alternativas.")


def robustez():
    """Resolve o modelo sob leituras alternativas das restricoes."""
    print("\nANALISE DE ROBUSTEZ DA FORMULACAO")
    print(f"{'Leitura das restricoes':<44}{'Receita (US$)':>16}")
    print("-" * 60)
    casos = [
        ("teor exato, reserva como limite superior",  True,  False),
        ("teor minimo, reserva como limite superior", False, False),
        ("teor exato, toda a reserva lavrada",        True,  True),
        ("teor minimo, toda a reserva lavrada",       False, True),
    ]
    receitas = []
    for rotulo, te, ct in casos:
        s = resolver(rotulo, teor_exato=te, consumir_tudo=ct)
        receitas.append(s.receita)
        print(f"{rotulo:<44}{s.receita:>16,.0f}".replace(",", "."))
    print("-" * 60)
    if max(receitas) - min(receitas) < 1e-4:
        print("As quatro leituras conduzem a mesma receita otima.")
    else:
        print(f"Diferenca entre leituras: "
              f"US$ {max(receitas)-min(receitas):,.2f}".replace(",", "."))


def comparar_sem_blendagem(s):
    """Quantifica o ganho economico atribuivel a blendagem."""
    receita_ref, detalhe, menor = sem_blendagem()
    print("\nCOMPARACAO COM POLITICA SEM BLENDAGEM")
    print("Cada frente vendida pela faixa correspondente ao proprio teor,")
    print("arredondado para baixo; frentes abaixo de "
          f"{menor}% P2O5 nao tem valor comercial.\n")
    print(f"{'Frente':<14}{'Teor':>7}{'Faixa':>8}{'Receita (US$)':>16}")
    print("-" * 45)
    for i in NOMES:
        faixa, valor = detalhe[i]
        rot = f"{faixa}%" if faixa else "rejeito"
        print(f"{i:<14}{G[i]:>7.1f}{rot:>8}{valor:>16,.0f}".replace(",", "."))
    print("-" * 45)
    print(f"{'Sem blendagem':<29}{receita_ref:>16,.0f}".replace(",", "."))
    print(f"{'Com blendagem (otimo)':<29}{s.receita:>16,.0f}"
          .replace(",", "."))
    ganho = s.receita - receita_ref
    print(f"{'Ganho':<29}{ganho:>16,.0f}".replace(",", ".")
          + f"   ({100*ganho/receita_ref:+.2f}%)")


def gerar_figura(s, arquivo="blendagem.png"):
    """Figura 1: contribuicao de cada frente ao produto de 24%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pct = s.percentuais()
    M = s.massa_por_faixa()
    ativas = [j for j in FAIXAS if M[j] > 1e-6]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) composicao de cada produto
    largura = 0.6
    base = {j: 0.0 for j in ativas}
    for i in NOMES:
        alturas = [pct[(i, j)] for j in ativas]
        ax[0].bar([f"{j}%" for j in ativas], alturas, largura,
                  bottom=[base[j] for j in ativas], label=i)
        for j, h in zip(ativas, alturas):
            base[j] += h
    ax[0].set_ylabel("Participacao no produto (%)")
    ax[0].set_xlabel("Faixa comercial")
    ax[0].set_title("(a) Composicao de cada produto")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(axis="y", alpha=0.3)

    # (b) destino da reserva de cada frente
    idx = range(len(NOMES))
    acum = [0.0] * len(NOMES)
    for j in ativas:
        vals = [s.x[(i, j)] / 1000 for i in NOMES]
        ax[1].barh(list(idx), vals, left=acum, label=f"produto {j}%")
        acum = [a + v for a, v in zip(acum, vals)]
    ax[1].set_yticks(list(idx))
    ax[1].set_yticklabels(NOMES, fontsize=8)
    ax[1].set_xlabel("Massa alocada (mil t)")
    ax[1].set_title("(b) Destino da reserva de cada frente")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    print(f"\nFigura salva em {arquivo}")


# =====================================================================
# 5. EXECUCAO
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figura", action="store_true",
                    help="gera a figura dos resultados")
    ap.add_argument("--sem-blendagem", action="store_true",
                    help="compara com a politica de referencia sem mistura")
    ap.add_argument("--sem-desempate", action="store_true",
                    help="nao aplica o criterio de desempate entre solucoes "
                         "otimas alternativas")
    args = ap.parse_args()

    print("=" * 62)
    print("BLENDAGEM DE MINERIO DE FOSFATO POR PROGRAMACAO LINEAR")
    print("Reproducao de Ali et al. (2018), Min. Miner. Depos. 12(4), 1-8")
    print("=" * 62)

    imprimir_dados()

    s = resolver("base", desempate=not args.sem_desempate)
    imprimir_solucao(s)
    validar(s)
    demonstrar_degenerescencia()
    robustez()

    if args.sem_blendagem:
        comparar_sem_blendagem(s)
    if args.figura:
        gerar_figura(s)


if __name__ == "__main__":
    main()