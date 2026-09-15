# 5.4 Resultados

Nesta secao, analisa-se criticamente a transicao do vies algoritmico **a nivel de dados (*pre-treinamento*)** para o vies manifestado nas **predicoes dos modelos de aprendizado de maquina (*pos-treinamento*)**, culminando na avaliacao empirica da **mitigacao via Redes Neurais com penalizacao de vies (*FairMLP com variacao de lambda*)**. A investigacao organiza-se em tres subsecoes dedicadas a cada *dataset* do escopo: **Adult (US Census)**, **COMPAS (Justica Criminal)** e **SINASC (DATASUS, Saude Neonatal no Brasil)**.

O objetivo central desta analise empirica e demonstrar como o treinamento supervisionado padrao (*baselines*) reage aos desequilibrios historicos dos dados brutos --- ora **amplificando desproporcionalmente a penalizacao interseccional**, ora **gerando ilusoes estatisticas de equidade sob desbalanceamento severo** ---, e como a introducao preliminar de penalizacao em uma Rede Neural Artificial (RNA) comprova a viabilidade da mitigacao, mas ao mesmo tempo expoe os limites da parametrizacao escalar fixa, fundamentando a arquitetura metodologica proposta para o doutorado (**Rede Neural Multitarefa com Camada Reversa de Gradiente + Otimizacao Multiobjetivo NSGA-II + Interpretabilidade por SHAP**).

---

## 5.4.1 Caso 1: Adult (US Census) --- Renda, Amplificacao e Mitigacao via RNA

### A. Diagnostico do Vies nos Dados Brutos (Pre-Treinamento)

O *dataset* Adult manteve 42.599 (87%) das 48.842 instancias originais apos o pre-processamento. Os dados apresentam forte desbalanceamento da classe-alvo, com apenas cerca de 25% das amostras atingindo o criterio de alta renda (renda anual superior a US$50K). A escassez de exemplos positivos pode prejudicar o aprendizado de padroes para os desfechos favoraveis. Alem disso, representa uma dificuldade adicional para minorias com baixa representatividade, que consequentemente tambem tendem a apresentar pouca interseccao com a classe positiva.

Os atributos sensiveis e *proxies* socioeconomicos em geral sao multiclasse no Adult. Para avaliacao de equidade de forma unidimensional em nivel de dados, adotamos como criterio atribuir os grupos com maior taxa de sucesso (renda anual superior a US$50K) como privilegiados e aqueles com menor taxa como desprivilegiados.

| Atributo | Privilegiado | Desprivilegiado | CI | DI | KL | KS |
|---|---|---|---|---|---|---|
| sex | Male | Female | 0.35 | 0.37 | 0.141 | 0.200 |
| race | White | Black | **0.80** | 0.47 | 0.072 | 0.141 |
| age_group | Middle-aged | Young | -0.10 | 0.34 | 0.188 | 0.240 |
| education_group | Graduate Degree | Schooling | -0.68 | 0.23 | 0.614 | **0.483** |
| relationship | Wife | Own-child | -0.47 | **0.03** | **1.287** | 0.466 |

**Tabela 5.X --- Metricas dinamicas de vies avaliadas por atributo sensivel (Adult).**
*CI: Class Imbalance | DI: Disparate Impact | KL: Kullback-Leibler | KS: Kolmogorov-Smirnov*

Com relacao a distribuicao de amostras, a metrica CI mostra que o *dataset* apresenta um numero consideravelmente maior de exemplos de populacoes privilegiadas para os atributos protegidos sexo e raca. Ao mesmo tempo, enquanto o atributo idade mostra uma distribuicao mais equilibrada, o conjunto tem um tamanho amostral muito superior para grupos desprivilegiados nos *proxies* socioeconomicos educacao e relacionamento.

Quanto as taxas de sucesso, o DI pre-treino mostra grandes disparidades na distribuicao entre as populacoes privilegiada e desprivilegiada em todos os tres atributos sensiveis, assumindo valores entre 0,34 e 0,47. Em relacao a sexo e raca, o privilegio nas taxas de sucesso de homens brancos fica evidente.

Em relacao a idade, o fator renda pode se confundir com a progressao de carreira e construcao de patrimonio ao longo da vida. Os *proxies* socioeconomicos tambem podem apresentar fatores de confusao similares: maior educacao tende a atrair oportunidades para melhores salarios, ao mesmo tempo que pessoas casadas tendem a compor renda de forma conjunta na unidade familiar. Contudo, estas variaveis tambem representam acesso a educacao e padroes de relacionamento influenciados por condicoes socioeconomicas, especialmente no contexto onde os dados foram coletados, na sociedade estadunidense da decada de 1990.

Neste cenario, nao se pode ignorar que estas variaveis apresentam disparidades ainda maiores que os atributos sensiveis, com DI de 0,23 ao comparar pessoas com educacao basica aquelas com pos-graduacao, e 0,03 ao comparar mulheres casadas com pessoas que criam filhos sozinhas. Neste ultimo exemplo, a metrica encontrada mostra uma taxa de sucesso quase inexistente para a populacao desprivilegiada.

A divergencia KL e o Kolmogorov-Smirnov confirmaram a diferenca de distribuicao de positivos, com divergencias menores porem significativas nos grupos segmentados pelos atributos sensiveis, e bem maiores nos *proxies* socioeconomicos. Estas metricas mostram o vies no nivel de dados, que reflete as desigualdades estruturais nos niveis de renda do recorte populacional que constitui o *dataset*.

**Auditoria Interseccional de Gerrymandering:**

| Par de Interseccao | Gap A | Gap B | Esperado | Real | Oculto |
|---|---|---|---|---|---|
| **education_group x relationship** | **48,28%** | **46,58%** | **48,28%** | **83,85%** | **+35,57%** |
| **sex x education_group** | **20,02%** | **48,28%** | **48,28%** | **65,72%** | **+17,44%** |
| **age_group x education_group** | **23,98%** | **48,28%** | **48,28%** | **63,07%** | **+14,78%** |
| **sex x age_group** | **20,02%** | **23,98%** | **23,98%** | **37,13%** | **+13,16%** |
| **age_group x relationship** | **23,98%** | **46,58%** | **46,58%** | **57,83%** | **+11,25%** |
| race x age_group | 14,11% | 23,98% | 23,98% | 33,10% | +9,12% |
| race x education_group | 14,11% | 48,28% | 48,28% | 56,40% | +8,12% |
| sex x race | 20,02% | 14,11% | 20,02% | 26,74% | +6,71% |
| race x relationship | 14,11% | 46,58% | 46,58% | 49,60% | +3,02% |
| sex x relationship | 20,02% | 46,58% | 46,58% | 46,85% | +0,27% |

**Tabela 5.Y --- Auditoria de vies interseccional oculto em pares de atributos (Adult).**
*Gap Esperado e o valor maximo entre Gap A e Gap B. Em **negrito**, interseccoes com Vies Oculto > 10% (limiar de gerrymandering = 0,05%).*

Metade das interseccoes mostram vies oculto severo, com disparidade nas taxas crescendo acima de 10% na analise cruzada em relacao a analise marginal. E uma demonstracao clara do efeito combinatorio do vies, que reforca que a analise marginal fica incompleta, pois perde informacao relevante sobre as penalizacoes sobrepostas. O atributo educacao, selecionado como *proxy* socioeconomico, atua como forte catalisador das disparidades, mostrando as maiores taxas de adicao de desigualdade quando cruzado a todo atributo sensivel ou *proxy* socioeconomico.

Observando os atributos sensiveis, embora a literatura aborde mais frequentemente o vies racial, as interseccoes incluindo sexo ou idade apresentaram maior vies oculto. Os cruzamentos com raca nao ultrapassaram o limiar de 10%. Isso pode indicar que o vies racial presente neste *dataset* se manifesta de forma consistente em todo o grupo marginalmente --- ou que variaveis de classe socioeconomica (como educacao) estao absorvendo e mascarando o impacto racial no nivel de dados, atuando como variaveis de confusao.

**Analise CDDL --- Disparidade Condicionada por Proxy:**

| Atributo Protegido (Vitima) | Grupo Desprivilegiado | Variavel Proxy (Condicao) | CDDL |
|---|---|---|---|
| age_group | Young | race | **0,3106** |
| age_group | Young | education_group | 0,3043 |
| education_group | Schooling | relationship | 0,2967 |
| age_group | Young | sex | 0,2956 |
| education_group | Schooling | sex | 0,2712 |
| education_group | Schooling | age_group | 0,2652 |
| education_group | Schooling | race | 0,2511 |
| sex | Female | education_group | 0,2470 |
| sex | Female | race | 0,2240 |
| age_group | Young | relationship | 0,2218 |
| sex | Female | age_group | 0,2188 |
| relationship | Own-child | race | 0,1685 |
| relationship | Own-child | education_group | 0,1636 |
| relationship | Own-child | sex | 0,1634 |
| relationship | Own-child | age_group | 0,1450 |
| race | Black | age_group | 0,0713 |
| sex | Female | relationship | 0,0691 |
| race | Black | education_group | 0,0590 |
| race | Black | sex | 0,0579 |
| race | Black | relationship | 0,0516 |

**Tabela 5.C --- Matriz completa de CDDL por atributo protegido (vitima) x variavel proxy (condicao), Adult.**
*Ordenada por CDDL decrescente. Cada linha mede a disparidade condicional sofrida pelo grupo desprivilegiado da coluna 2, quando os rotulos sao segmentados pelos estratos da variavel proxy da coluna 3.*

A avaliacao da metrica CDDL no *dataset* Adult Income revela que as disparidades nos rotulos nao se distribuem de maneira uniforme entre os atributos sensiveis, evidenciando uma forte dinamica interseccional. Os resultados demonstram que o vies afeta desproporcionalmente individuos jovens (*age_group = Young*) e com menor nivel de escolaridade formal (*education_group = Schooling*), que ocupam as tres primeiras posicoes do ranking. O grupo de jovens apresentou o maior indice de disparidade de todo o experimento (CDDL = 0,310) quando o atributo raca foi introduzido como variavel de condicionamento (*proxy*), sugerindo que a desvantagem estrutural associada a idade e severamente amplificada ao se considerar recortes raciais.

Adicionalmente, a raca atua como o *proxy* mais forte para expor disparidades latentes em outros grupos demograficos. Quando *race* e utilizada como condicao, o CDDL medio das vitimas atinge 0,238, o maior impacto entre todas as variaveis condicionantes analisadas.

Um achado contraintuitivo desta analise e o comportamento do proprio atributo *race* quando tratado como atributo protegido vitima: individuos negros apresentaram os menores valores de CDDL (entre 0,051 e 0,071), independentemente da variavel de condicionamento aplicada. Este fenomeno indica que, para este cenario especifico, as disparidades focadas estritamente em raca parecem mitigadas ou mascaradas por outras variaveis estruturais --- exigindo uma investigacao mais profunda sobre como o vies de representacao historica esta sendo codificado nos dados originais.

**Nota metodologica sobre o recorte interseccional:** diferentemente da auditoria de dados acima --- que avalia os cinco atributos sensiveis disponiveis (sex, race, age_group, education_group, relationship) e todos os seus pares ---, o treinamento dos modelos e a avaliacao pos-treino das Secoes B e C abaixo foram restritos a uma unica interseccao por *dataset* (sex x race no Adult e no COMPAS; raca_cor_mae x idade_mae no SINASC). A escolha se justifica por dois motivos: (i) comparabilidade direta com a literatura de justica algoritmica, que tipicamente usa sex x race como recorte padrao no Adult; e (ii) viabilidade computacional do pipeline completo (validacao cruzada aninhada + lambda sweep da FairMLP) dentro do escopo desta proposta de qualificacao. E importante registrar que a propria auditoria de gerrymandering (Tabela 5.Y) identificou pares com vies oculto ainda maior que sex x race --- notavelmente education_group x relationship (+35,57%, o maior de toda a tabela) ---, que nao foram submetidos ao treinamento de modelos nesta etapa. A extensao da avaliacao pos-treino e da mitigacao para multiplos atributos sensiveis simultaneos, incluindo pares alem do escolhido aqui, e endereçada diretamente pela arquitetura multitarefa com Camada Reversa de Gradiente proposta para a tese (Secao 5.4.4), desenhada desde a concepcao para mais de dois atributos sensiveis ao mesmo tempo --- superando a limitacao do recorte bivariado utilizado nestes experimentos preliminares.

---

### B. Efeitos do Treinamento nos Baselines Tradicionais

Ao decompor a base pelos atributos sensiveis Genero e Raca, emergem disparidades substanciais de paridade estatistica na distribuicao original:

* **Male & White** (N = 26.657): 32,79% possuem renda favoravel (grupo de referencia, Pre-DI = 1,000).
* **Male & Black** (N = 2.198): 19,02% possuem renda favoravel (Pre-DI = 0,580; Pre-SPD = -13,77%).
* **Female & White** (N = 11.595): 12,76% possuem renda favoravel (Pre-DI = 0,389; Pre-SPD = -20,02%).
* **Female & Black** (N = 2.149): apenas 6,05% possuem renda favoravel (Pre-DI = 0,185; Pre-SPD = -26,74%).

O melhor *baseline* em termos de utilidade preditiva foi o Gradient Boosting otimizado por PR-AUC, atingindo acuracia de 86,5% e ROC-AUC de 0,919. Todavia, o treinamento **amplificou** a penalizacao sobre o grupo mais vulneravel: a taxa de predicao favoravel para mulheres negras caiu de 6,05% (dados) para apenas 4,80% (modelo), derrubando o Disparate Impact de Pre-DI = 0,185 para Post-DI = 0,174. A TPR deste subgrupo atingiu apenas 56,1% e o Max Intersectional AAOD chegou a 0,0845.

| Subgrupo | N (teste/fold) | Taxa Favoravel | Post-DI | TPR | FPR | AAOD |
|---|---|---|---|---|---|---|
| Male & White | 8.886 | 0,275 | 1,000 | 0,655 | 0,089 | 0,000 |
| Male & Black | 733 | 0,142 | 0,516 | 0,586 | 0,037 | 0,061 |
| Female & White | 3.865 | 0,098 | 0,357 | 0,601 | 0,025 | 0,059 |
| **Female & Black** | **716** | **0,048** | **0,174** | **0,561** | **0,014** | **0,084** |

**Tabela 5.Z --- Metricas de equidade interseccional pos-treino por subgrupo (Adult, GradientBoosting --- melhor baseline, media entre 3 folds externos).**

O padrao observado e paradigmatico: o modelo aprende a descartar precocemente instancias de mulheres negras por sua baixissima representacao entre positivos, gerando uma taxa de falsos negativos elevada (FNR = 43,9%) que sequer se reflete no AUROC global, o qual permanece confortavelmente acima de 0,91.

---

### C. Mitigacao por Rede Neural Artificial (FairMLP: Variacao de lambda)

Para avaliar a viabilidade de mitigar a penalizacao interseccional, treinou-se uma FairMLP sob variacao do escalar de penalizacao lambda em {0,0; 0,5; 1,0; 2,0; 4,0}. Como a penalizacao de equidade e ponderada por lambda na funcao de perda (*loss = BCE + lambda x penalidade_dp*), a configuracao lambda = 0,0 anula completamente esse termo, funcionando como uma **MLP convencional sem qualquer mitigacao de vies** --- o baseline de rede neural equivalente aos baselines de arvore (Random Forest/Gradient Boosting) da Secao B, mas mantendo a mesma arquitetura da FairMLP:

| Configuracao | Acuracia | ROC-AUC | PR-AUC | Max AAOD | Sensitivity Gap |
|---|---|---|---|---|---|
| **MLP sem mitigacao** (lambda = 0,0) | 0,8485 +/- 0,0010 | 0,9034 +/- 0,0012 | 0,7789 +/- 0,0003 | 0,0940 +/- 0,0344 | 0,1025 +/- 0,0476 |
| FairMLP (lambda = 0,5) | 0,8483 +/- 0,0011 | 0,9033 +/- 0,0014 | 0,7786 +/- 0,0007 | 0,0804 +/- 0,0313 | 0,1018 +/- 0,0251 |
| FairMLP (lambda = 1,0) | 0,8480 +/- 0,0014 | 0,9029 +/- 0,0016 | 0,7780 +/- 0,0011 | 0,0789 +/- 0,0259 | 0,1079 +/- 0,0166 |
| FairMLP (lambda = 2,0) | 0,8473 +/- 0,0004 | 0,9015 +/- 0,0023 | 0,7759 +/- 0,0026 | **0,0668 +/- 0,0027** | 0,1212 +/- 0,0346 |
| FairMLP (lambda = 4,0) | 0,8442 +/- 0,0012 | 0,8965 +/- 0,0039 | 0,7695 +/- 0,0046 | 0,0692 +/- 0,0116 | 0,1571 +/- 0,0320 |

**Tabela 5.W --- Adult: Efeito da penalizacao de vies na FairMLP (lambda sweep).**

1. **Reducao Substancial da Injustica:** A introducao do fator lambda = 2,0 reduziu o Max AAOD de 0,0940 para 0,0668, uma reducao de **29,0% na disparidade maxima**, com perda residual desprezivel na acuracia (queda de aproximadamente 0,12 ponto percentual) e no ROC-AUC (0,9034 para 0,9015).
2. **Ponto de Inflexao e Degradacao:** Ao aumentar para lambda = 4,0, o modelo atinge saturacao: o Recall desaba de 62,1% para 54,0% e o Sensitivity Gap sobe para 0,1571, demonstrando empiricamente que a relacao entre equidade e penalizacao e **nao-monotona**, exigindo mecanismos avancados de busca de Pareto.

---

## 5.4.2 Caso 2: COMPAS (Justica Penal) --- Risco, Assimetria e Mitigacao via RNA

### A. Diagnostico do Vies nos Dados Brutos (Pre-Treinamento)

No COMPAS (N = 6.688), a condicao favoravel e a **nao-reincidencia em dois anos** (Y = 0, prevalencia global de 54,04%). O grupo com maior taxa de nao-reincidencia (Mulheres Caucasianas, 64,59%) foi tomado como referencia.

| Subgrupo | N | Taxa Favoravel | Pre-DI | Pre-SPD |
|---|---|---|---|---|
| Female & Caucasian | 562 | 0,6459 | 1,000 | 0,000 |
| Female & Hispanic | 102 | 0,6765 | 1,047 | +3,06% |
| Female & African-American | 646 | 0,6192 | 0,959 | -2,67% |
| Male & Caucasian | 1.832 | 0,5884 | 0,911 | -5,75% |
| Male & Hispanic | 526 | 0,6255 | 0,968 | -2,04% |
| **Male & African-American** | **3.020** | **0,4553** | **0,705** | **-19,06%** |

**Tabela 5.X2 --- Disparidades de paridade estatistica pre-treino por subgrupo interseccional (COMPAS).**

A magnitude da disparidade entre os grupos extremos e notoria: a taxa de nao-reincidencia de homens negros (45,53%) esta **19,06 pontos percentuais** abaixo do grupo de referencia, com Pre-DI de 0,705 --- abaixo do limiar da Regra dos 80%, configurando potencial impacto desproporcional nos dados brutos antes mesmo do treinamento. Homens negros constituem 45,2% da base, sendo o maior subgrupo, ao passo que os demais concentram-se em faixas muito menores.

---

### B. Efeitos do Treinamento nos Baselines Tradicionais

O melhor *baseline* (Gradient Boosting, otimizado por ROC-AUC) atingiu acuracia de 64,1% e ROC-AUC de 0,691. Os *baselines* **acentuaram severamente** as assimetrias de erro contra reus negros:

| Subgrupo | N (teste/fold) | Taxa Favoravel | Post-DI | TPR | FPR | AAOD |
|---|---|---|---|---|---|---|
| Female & Caucasian | 187 | 0,710 | 0,950 (ref.)* | 0,804 | 0,538 | 0,033 |
| Female & Hispanic | 34 | 0,724 | 0,961 | 0,871 | 0,435 | 0,072 |
| Female & African-American | 215 | 0,567 | 0,759 | 0,671 | 0,399 | 0,170 |
| Male & Caucasian | 611 | 0,634 | 0,849 | 0,733 | 0,494 | 0,091 |
| Male & Hispanic | 175 | 0,657 | 0,879 | 0,736 | 0,523 | 0,079 |
| **Male & African-American** | **1.007** | **0,437** | **0,583** | **0,573** | **0,323** | **0,256** |

**Tabela 5.Y2 --- Metricas de equidade interseccional pos-treino por subgrupo (COMPAS, GradientBoosting --- melhor baseline, media entre 3 folds externos).**
*Female & Caucasian foi o grupo de referencia (maior taxa favoravel) em 2 dos 3 folds externos; no fold restante, Female & Hispanic assumiu a referencia com N=29 (abaixo do limiar de viabilidade de N=30), por isso o Post-DI medio nao chega exatamente a 1,000.

A TPR entre reus afro-americanos do sexo masculino (57,3%) ficou **23,1 pontos percentuais** abaixo da TPR de mulheres caucasianas (80,4%), resultando em Max Intersectional AAOD de 0,256. O modelo aprende que a condicao de ser homem negro e preditora de reincidencia, amplificando a taxa de falsos positivos especificamente neste grupo --- uma manifestacao algoritmica direta do padrao documentado por Angwin et al. (2016).

---

### C. Mitigacao por Rede Neural Artificial (FairMLP: Variacao de lambda)

Como no Caso 1, a linha lambda = 0,0 corresponde a **MLP sem mitigacao** (penalizacao de equidade anulada), servindo de baseline de rede neural comparavel ao Gradient Boosting da Secao B:

| Configuracao | Acuracia | ROC-AUC | PR-AUC | Max AAOD | Sensitivity Gap |
|---|---|---|---|---|---|
| **MLP sem mitigacao** (lambda = 0,0) | 0,6361 +/- 0,0080 | 0,6902 +/- 0,0061 | 0,6972 +/- 0,0109 | 0,2649 +/- 0,0548 | 0,2298 +/- 0,0288 |
| FairMLP (lambda = 0,5) | 0,6361 +/- 0,0080 | 0,6902 +/- 0,0061 | 0,6972 +/- 0,0109 | 0,2649 +/- 0,0548 | 0,2298 +/- 0,0288 |
| FairMLP (lambda = 1,0) | 0,6361 +/- 0,0080 | 0,6901 +/- 0,0062 | 0,6968 +/- 0,0114 | 0,2649 +/- 0,0548 | 0,2298 +/- 0,0288 |
| FairMLP (lambda = 2,0) | 0,6343 +/- 0,0065 | 0,6889 +/- 0,0077 | 0,6952 +/- 0,0143 | 0,2666 +/- 0,0532 | 0,2233 +/- 0,0230 |
| FairMLP (lambda = 4,0) | 0,6346 +/- 0,0069 | 0,6882 +/- 0,0058 | 0,6938 +/- 0,0103 | **0,2284 +/- 0,0130** | **0,1972 +/- 0,0349** |

**Tabela 5.Z2 --- COMPAS: Efeito da penalizacao de vies na FairMLP (lambda sweep).**

Para lambda em {0,0; 0,5; 1,0}, a RNA converge para exatamente o mesmo patamar de AAOD (0,2649) que os *baselines* tradicionais --- sem qualquer avanco de equidade. Somente com lambda = 4,0 observou-se reducao relevante: Max AAOD de 0,2649 para 0,2284 (-13,8%) e Sensitivity Gap de 0,2298 para 0,1972 (-14,2%).

O contraste com o Adult e revelador: enquanto no Adult lambda = 2,0 ja produzia ganhos de equidade substanciais, no COMPAS a mesma magnitude nao produz efeito. O vies no COMPAS e **mais profundamente enraizado nos padroes de dados** --- a disparidade estrutural de 19% entre grupos na base bruta cria gradientes de aprendizado que resistem a regularizacao escalar, fundamentando a necessidade de estrategias de mitigacao mais sofisticadas, como o NSGA-II com otimizacao multiobjetivo.

---

## 5.4.3 Caso 3: SINASC (DATASUS) --- Ilusao de Equidade sob Desbalanceamento Severo

### A. Diagnostico do Vies nos Dados Brutos (Pre-Treinamento)

O SINASC (N = 2.474.535) apresenta o mais severo desbalanceamento de classes do conjunto de experimentos: 90,57% dos nascimentos apresentam peso normal (desfecho favoravel, Y = 1), contra apenas 9,43% com baixo peso (Y = 0, desfecho desfavoravel). Esta assimetria extrema cria condicoes propicias para classificadores triviais (*dummy classifiers*) que predizem sempre a classe majoritaria com altissima acuracia nominal.

| Subgrupo | N | Taxa Favoravel | Pre-DI | Pre-SPD |
|---|---|---|---|---|
| Branca & Adulta (20-34) | 588.856 | 0,9131 | 1,000 (ref.) | 0,000 |
| Amarela & Adulta (20-34) | 7.753 | 0,9138 | 1,001 | +0,07% |
| Indigena & Adulta (20-34) | 16.311 | 0,9166 | 1,004 | +0,35% |
| Parda & Adulta (20-34) | 991.802 | 0,9136 | 1,000 | +0,05% |
| Preta & Adulta (20-34) | 130.078 | 0,8970 | 0,982 | -1,61% |
| Branca & Jovem (<20) | 68.384 | 0,8936 | 0,979 | -1,95% |
| Indigena & Jovem (<20) | 7.138 | 0,8784 | 0,962 | -3,47% |
| Preta & Jovem (<20) | 20.366 | 0,8788 | 0,962 | -3,43% |
| Parda & Jovem (<20) | 207.038 | 0,8938 | 0,979 | -1,93% |
| Preta & Madura (>34) | 31.961 | 0,8709 | 0,954 | **-4,22%** |
| Indigena & Madura (>34) | 2.852 | 0,9092 | 0,996 | -0,39% |

**Tabela 5.X3 --- Selecao de disparidades de paridade estatistica pre-treino por subgrupo (SINASC). Referencia: Branca & Adulta (20-34).**

A primeira vista, as disparidades de Pre-SPD parecem modestas (entre -4,22% e +0,35%). Contudo, esta aparente moderacao e **estatisticamente enganosa**: uma diferenca de 4% na taxa de desfecho favoravel, partindo de uma base de 90%, representa que maes pretas maduras tem uma probabilidade de **baixo peso ao nascer quase 50% maior** que maes brancas adultas. O subgrupo de maior vulnerabilidade combinada e **Preta & Madura (>34)** com Pre-SPD de -4,22%.

O padrao interseccional e clinicamente coerente com a literatura obstetrica: mulheres mais jovens (abaixo de 20 anos) e mais maduras (acima de 34 anos) apresentam consistentemente maiores taxas de baixo peso ao nascer em todas as racas/cores, caracterizando os estratos de risco extremo de idade materna. Maes de raca/cor Preta apresentam as menores taxas favoraveis dentro de cada faixa etaria.

---

### B. Efeitos do Treinamento nos Baselines Tradicionais e Ilusao de Equidade

Os resultados do SINASC revelam o fenomeno mais critico dos tres experimentos. Random Forest e Gradient Boosting atingiram acuracia de 90,57% e PR-AUC de 0,914, com Recall de 100%, Max Intersectional AAOD de 0,000 e Sensitivity Gap de 0,000 --- aparentemente **equidade perfeita**.

Esta e uma **ilusao estatistica perigosa**. Os modelos aprenderam a predizer **sempre a classe majoritaria (peso normal)**, sem detectar um unico caso de baixo peso ao nascer. O ROC-AUC de apenas 0,532 (marginalmente acima da linha de chance de 0,500) confirma que os modelos nao possuem capacidade discriminativa real. O Recall de 100% reflete que todas as instancias preditas sao positivas --- nao ha nenhuma predicao negativa.

O resultado de AAOD = 0 e Sensitivity Gap = 0 e consequencia direta desse comportamento: quando todos os subgrupos recebem apenas predicoes positivas, as taxas de erro sao identicamente zero para todos. Esta **paridade trivial** e a manifestacao mais insidiosa do vies algoritmico sob desbalanceamento severo: as metricas de equidade tradicionais reportam conformidade inexistente.

---

### C. Mitigacao por Rede Neural Artificial (FairMLP: Variacao de lambda)

Novamente, lambda = 0,0 representa a **MLP sem mitigacao**, reproduzindo o mesmo colapso trivial ja observado nos baselines de arvore da Secao B:

| Configuracao | Acuracia | ROC-AUC | PR-AUC | Max AAOD | Sensitivity Gap |
|---|---|---|---|---|---|
| **MLP sem mitigacao** (lambda = 0,0) | 0,9057 +/- 0,0000 | 0,5326 +/- 0,0006 | 0,9138 +/- 0,0002 | 0,000 | 0,000 |
| FairMLP (lambda = 0,5) | 0,9057 +/- 0,0000 | 0,5323 +/- 0,0008 | 0,9137 +/- 0,0002 | 0,000 | 0,000 |
| FairMLP (lambda = 1,0) | 0,9057 +/- 0,0000 | 0,5326 +/- 0,0006 | 0,9138 +/- 0,0002 | 0,000 | 0,000 |
| FairMLP (lambda = 2,0) | 0,9057 +/- 0,0000 | 0,5324 +/- 0,0009 | 0,9138 +/- 0,0002 | 0,000 | 0,000 |
| FairMLP (lambda = 4,0) | 0,9057 +/- 0,0000 | 0,5323 +/- 0,0010 | 0,9137 +/- 0,0003 | 0,000 | 0,000 |

**Tabela 5.Z3 --- SINASC: Efeito da penalizacao de vies na FairMLP (lambda sweep).**

A FairMLP reproduz exatamente o mesmo comportamento colapsado dos *baselines* ao longo de todo o espectro de lambda. A penalizacao de equidade escalar nao tem qualquer efeito quando o modelo ja atingiu o minimo trivial de perda ao predizer a classe majoritaria. A funcao L_fairness nao gera gradientes informativos, pois todos os subgrupos recebem o mesmo desfecho predito.

Esta e a limitacao fundamental que a presente tese busca superar: **a penalizacao escalar de equidade e ineficaz quando o problema de desbalanceamento de classes nao e tratado conjuntamente**. A resolucao exige uma abordagem de otimizacao multiobjetivo que simultaneamente considere a penalizacao de classe e a penalizacao de equidade --- exatamente o que a arquitetura NSGA-II proposta endereça.

---

## 5.4.4 Sintese Comparativa e Fundamentacao da Proposta Metodologica

| Dataset | Cenario de Falha | Manifestacao | Efeito da FairMLP |
|---|---|---|---|
| **Adult** | Amplificacao de vies moderado | Post-DI = 0,174 para Female & Black; AAOD = 0,085 | Mitigacao parcial efetiva: -29,0% AAOD com lambda = 2,0 |
| **COMPAS** | Amplificacao de vies severo e resistente | AAOD = 0,256; Sensitivity Gap = 23,1pp | Resistencia estrutural: apenas lambda = 4,0 produz -13,8% AAOD |
| **SINASC** | Ilusao de equidade por colapso trivial | AAOD = 0 por predicao uniforme; ROC-AUC = 0,532 | Nenhum efeito em qualquer lambda; FairMLP colapsa identicamente |

**Tabela 5.S --- Sintese comparativa dos tres cenarios experimentais.**

Este conjunto de achados fundamenta empiricamente a proposta metodologica da tese em tres dimensoes:

1. **Necessidade de otimizacao multiobjetivo (NSGA-II):** A relacao entre equidade e utilidade preditiva e nao-monotona (Adult) e resistente a penalizacao escalar (COMPAS), tornando a busca de compromissos otimos de Pareto indispensavel.
2. **Necessidade de mitigacao integrada de desbalanceamento:** O colapso total no SINASC demonstra que a regularizacao de equidade e ineficaz sem um componente paralelo que endereça a assimetria de classes --- seja via ponderacao de amostras, *oversampling* focado, ou funcao de perda composta.
3. **Necessidade de metricas auditaveis e interseccionais:** A aparente equidade perfeita do SINASC e o ROC-AUC favoravel do Adult seriam suficientes para validar estes modelos em protocolos de avaliacao tradicionais, demonstrando que a auditoria interseccional com metricas de pior caso (Max AAOD, Sensitivity Gap, PR-AUC) e condicao necessaria, nao opcional, para sistemas de decisao algoritmica em contextos de alto impacto social.
