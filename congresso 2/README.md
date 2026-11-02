# Congresso 2 — Alocação de gás de injeção em um bloco de poços de gas lift

Código do resumo estendido da disciplina **Métodos de Otimização (210143)**, UFJF.

O trabalho tem duas metades. A primeira reproduz o modelo de alocação econômica de
Zhong et al., *Energies* **15**(19):6950, 2022. A segunda é extensão própria, e
acrescenta a depleção do reservatório, o que acopla as decisões no tempo e muda a
classe do problema.

---

## Como os módulos se encaixam

```
gaslift.py                    núcleo: dados, curvas, solvers do problema estático
   │
   ├──> deplecao.py           extensão temporal (importa gaslift, não o modifica)
   │       │
   │       └──> figuras_deplecao.py    Figuras 2 e 3 do resumo
   │
   └──> figuras.py            figuras do problema estático
          
malha.py                      importa gaslift E deplecao; refino da malha temporal

verifica.py  reproduz.py  escala.py     scripts iniciais, autônomos
```

A regra que organiza tudo: **`gaslift.py` nunca é modificado**. A extensão da
Seção 5 importa dele e trabalha por cima, de modo que a reprodução do artigo
permanece intacta e auditável.

---

## O que cada arquivo faz

| Arquivo | Papel | Execução |
|---|---|---|
| `gaslift.py` | Núcleo. Dados das Tabelas 3 e 4, classe `Poco`, fator econômico $K_i$, e três solvers do problema estático. | `python gaslift.py` |
| `deplecao.py` | Extensão. Calibração das reservas, NLP multiperíodo no CasADi/IPOPT, enumeração dos cronogramas de fechamento e política míope. | `python deplecao.py` |
| `malha.py` | Refino da malha temporal, de 1 ano a 1 dia, e as quatro variantes de política comparadas. | `python malha.py 365 180 1` |
| `figuras.py` | Cinco figuras do problema estático. | `python figuras.py` |
| `figuras_deplecao.py` | Figuras 2 e 3 do resumo. | `python figuras_deplecao.py` |
| `verifica.py` | Confere os coeficientes derivados contra os publicados. Foi aqui que apareceu a divergência nos $\alpha_i$. | `python verifica.py` |
| `reproduz.py` | Alocação por KKT com bisseção, implementação independente. | `python reproduz.py` |
| `escala.py` | Custo computacional em instâncias sintéticas de tamanho crescente. | `python escala.py` |

---

## Detalhe dos módulos principais

### `gaslift.py`

Organizado em três camadas:

- **Camada 1, reprodução.** Das Tabelas 1 e 2 do artigo aos coeficientes da curva
  econômica, depois às fórmulas fechadas do caso irrestrito, e então ao modelo
  restrito com os quatro esquemas de alocação.
- **Camada 2, análise.** O ponto $Q_{max}=4$ do Esquema 1, onde a solução obtida
  domina a publicada, e o contraste entre ótimo técnico e ótimo econômico.
- **Camada 3, escalabilidade.** Custo do NLP contínuo e da enumeração $2^m$ dos
  poços de tipo $c$ e $d$.

Três solvers, que servem de verificação cruzada entre si:

| Função | Método |
|---|---|
| `resolve_kkt` | KKT com bisseção no multiplicador de Lagrange, forma fechada |
| `resolve_penalidade` | Penalidade mista com Newton-Raphson, que é o método do artigo |
| `resolve_sqp` | SQP do SciPy, referência externa |

A hessiana do subproblema é diagonal somada a uma matriz de posto unitário, e a
identidade de Sherman-Morrison a inverte sem fatoração.

### `deplecao.py`

| Função | O que faz |
|---|---|
| `calibra_reservas` | Obtém $N^{max}$ a partir da taxa de declínio, $N^{max} = 365\,q_o^0/D$ |
| `resolve_cronograma` | NLP multiperíodo para um cronograma fixo de fechamento |
| `resolve_multiperiodo` | Enumera os $(T+1)^2$ cronogramas e retém o melhor |
| `politica_miope` | Relevanta a curva a cada ano e resolve o estático, sem olhar adiante |

O estado é adimensionalizado, $x = N/N^{max}$, o que tornou o NLP dez vezes mais
rápido e resolveu uma falha errática de convergência.

### `malha.py`

As quatro variantes comparadas no resumo:

| Função | Decisão | Estado | Antecipa |
|---|---|---|---|
| `campo_multiperiodos` | passo livre | mesmo passo | não |
| `campo_estado_fino` | anual | passo fino | não |
| `multiperiodos` | passo livre | mesmo passo | sim |
| `multiperiodos_decisao_anual` | anual | passo fino | sim |

É o módulo que produz a tabela de refino da malha e os números da Tabela 6.

---

## Dois pontos de atenção

**As figuras `fig_tipos.png` e `fig_resultado.png` não saem sozinhas.** Elas são
definidas em `figuras.py` depois do bloco `if __name__ == "__main__"`, de modo que
`gerar_todas` não as inclui. Para gerá-las:

```python
from gaslift import monta_pocos
import figuras
pocos = monta_pocos()
figuras.fig_tipos(pocos)
figuras.fig_resultado(pocos)
```

**Os três scripts iniciais redeclaram os dados.** `verifica.py`, `reproduz.py` e
`escala.py` carregam a Tabela 1 dentro de si, em vez de importar `gaslift.py`,
porque foram escritos antes dele. Funcionam e têm valor de verificação
independente, mas a fonte canônica dos dados é o `gaslift.py`.

---

## Dependências

```
numpy  scipy  matplotlib  casadi>=3.8
```

O CasADi e o IPOPT são usados apenas em `deplecao.py` e `malha.py`. A reprodução
do artigo roda só com NumPy e SciPy.
