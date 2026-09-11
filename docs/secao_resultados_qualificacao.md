# 5.3 Resultados Preliminares e Auditoria de Equidade Interseccional

Nesta seção, apresentam-se os resultados empíricos da avaliação de desempenho preditivo e de equidade algorítmica interseccional obtidos com os modelos de referência (*baselines*). A análise estrutura-se a partir dos experimentos executados sobre três bases de dados com diferentes contextos sociodemográficos e epidemiológicos: **Adult (US Census)**, **COMPAS (Recidivismo Criminal)** e **SINASC (DATASUS, Saúde Neonatal no Brasil)**. 

Os experimentos foram conduzidos sob o protocolo de Validação Cruzada Aninhada (*Nested Cross-Validation*, 5 *folds* externos para estimação de generalização e 3 *folds* internos para calibração de hiperparâmetros), contemplando múltiplos objetivos de otimização ($\text{accuracy}$, $\text{recall}$, $\text{precision}$, $\text{roc\_auc}$ e $\text{pr\_auc}$) aplicados aos algoritmos *Random Forest* (RF) e *Gradient Boosting* (GB).

---

## 5.3.1 Desempenho Preditivo Global e Análise Comparativa

A Tabela 1 sintetiza o comportamento preditivo e os indicadores de disparidade interseccional agregados em nível de *dataset* e arquitetura. As métricas reportam a média e o desvio-padrão computados sobre os *folds* de teste externos ao longo de todas as funções objetivo avaliadas no *inner-CV*.

```
========================================================================================================================
Tabela 1: Desempenho preditivo e disparidade interseccional consolidada (Médias ± Desvios-Padrão).
========================================================================================================================
Dataset     Modelo              Acurácia         ROC-AUC          PR-AUC           Max AAOD (↓)     Sens. Gap (↓)
------------------------------------------------------------------------------------------------------------------------
Adult       Gradient Boosting   0.860 ± 0.008   0.915 ± 0.009   0.813 ± 0.018   0.0936 ± 0.0260   0.1247 ± 0.0547
Adult       Random Forest       0.843 ± 0.014   0.899 ± 0.010   0.778 ± 0.022   0.1155 ± 0.0196   0.1658 ± 0.0544
------------------------------------------------------------------------------------------------------------------------
COMPAS      Gradient Boosting   0.638 ± 0.003   0.691 ± 0.000   0.698 ± 0.000   0.2614 ± 0.0076   0.2303 ± 0.0055
COMPAS      Random Forest       0.641 ± 0.001   0.690 ± 0.000   0.697 ± 0.000   0.2644 ± 0.0075   0.2354 ± 0.0039
------------------------------------------------------------------------------------------------------------------------
SINASC      Gradient Boosting   0.906 ± 0.000   0.532 ± 0.002   0.913 ± 0.001   0.0000 ± 0.0000   0.0000 ± 0.0000
SINASC      Random Forest       0.906 ± 0.000   0.532 ± 0.000   0.914 ± 0.000   0.0000 ± 0.0000   0.0000 ± 0.0000
========================================================================================================================
Nota: (↓) indica métricas onde valores menores representam maior equidade interseccional.
```

A distribuição global das métricas de desempenho preditivo é ilustrada na **Figura 1**, comparando o comportamento dos modelos em termos de Acurácia e Revocação (*Recall*). Os *boxplots* em escala de cinza capturam a variabilidade induzida pelas diferentes métricas de otimização interna, enquanto os marcadores mapeiam conjuntamente a base de dados (código de cores) e o critério de seleção de hiperparâmetros (formato do marcador).

No *dataset* **Adult**, o algoritmo *Gradient Boosting* demonstrou superioridade consistente frente ao *Random Forest*, atingindo ROC-AUC médio de $0.915 \pm 0.009$ (com pico de $0.919$ quando otimizado por $\text{roc\_auc}$) e PR-AUC de $0.813 \pm 0.018$, contra $0.899 \pm 0.010$ e $0.778 \pm 0.022$ do *Random Forest*. Essa vantagem estende-se à métrica de acurácia média ($0.860$ vs $0.843$).

No *dataset* **COMPAS**, ambos os modelos apresentaram desempenho estatisticamente equivalente, com ROC-AUC estabilizado em torno de $0.691$ e acurácia próxima a $64\%$. Observa-se a baixa dispersão entre os *folds* e entre as funções de custo (desvios-padrão $\le 0.003$), refletindo o limite empírico de separabilidade linear e não-linear das variáveis disponíveis no cadastro penal de Broward County.

---

## 5.3.2 Impacto das Funções Objetivo de Otimização no Inner-CV

A análise comparativa entre as funções objetivo utilizadas na seleção interna de hiperparâmetros revelou que a escolha da métrica de calibração afeta diretamente a severidade do viés algorítmico, mesmo em modelos sem intervenção explícita de *fairness*.

No *dataset* **Adult**:
1. **Otimização por Acurácia e Recall:** Favoreceu o equilíbrio operacional do classificador, atingindo um *Max AAOD* significativamente menor: $0.0736$ (otimizado por acurácia) e $0.0813$ (otimizado por *recall*), com *Sensitivity Gap* restrito a $0.0873$ e $0.0982$, respectivamente.
2. **Otimização por Precisão:** Ao penalizar severamente os falsos positivos no *inner-CV*, o algoritmo adotou limiares conservadores de classificação, concentrando as predições favoráveis ($>50\text{k}$) exclusivamente em instâncias de alta certeza. Esse conservadorismo amplificou drasticamente a injustiça interseccional: o *Sensitivity Gap* explodiu para $0.2209$ no *Gradient Boosting* e $0.2163$ no *Random Forest*, com o *Max AAOD* saltando para $0.1389$.

No *dataset* **COMPAS**, a variação da função objetivo provocou variações no equilíbrio entre sensibilidade e especificidade, com a otimização por $\text{pr\_auc}$ alcançando a maior revocação global ($0.724$ no GB), porém preservando patamares elevados de disparidade de chances igualadas (*Max AAOD* de $0.2512$ a $0.2699$).

Esses achados corroboram a hipótese de que o ajuste de modelos baseado em métricas cegas de desempenho é insuficiente e pode agravar desproporcionalmente a penalização de subgrupos marginalizados.

---

## 5.3.3 Auditoria Interseccional Pós-Treinamento e Análise de Subgrupos

A **Figura 2** expõe a relação de compromisso (*trade-off*) entre a capacidade discriminativa global ($\text{ROC-AUC}$) e a máxima injustiça interseccional observada ($\text{Max AAOD}$). A métrica $\text{Max AAOD}$ define o pior desvio absoluto médio de chances igualadas entre qualquer subgrupo interseccional e o grupo de referência:

$$\text{Max AAOD} = \max_{g \in \mathcal{G}} \frac{1}{2} \left( |\text{TPR}_g - \text{TPR}_{\text{ref}}| + |\text{FPR}_g - \text{FPR}_{\text{ref}}| \right)$$

onde $\mathcal{G}$ denota o conjunto de subgrupos formados pela intersecção dos atributos protegidos $A \times S$.

A desagregação dos resultados nos subgrupos interseccionais revela disparidades profundas que permaneceriam ocultas sob avaliações univariadas (*marginal fairness*):

### 1. Adult Dataset: A Penalização Composta de Mulheres Negras
Na base Adult, o grupo de referência (*Male & White*, $N=8.884$) apresentou uma taxa de predição favorável de $24.8\%$, com $\text{TPR} = 59.3\%$ e $\text{FPR} = 7.9\%$. Em contraste:
- O subgrupo **Female & Black** ($N=717$) obteve uma taxa de predição favorável de apenas **$3.8\%$**, correspondendo a uma Razão de Impacto Disparatado pós-treino ($\text{Post-DI}$) de apenas **$0.149$** e $\text{AAOD} = 0.1020$. A taxa de verdadeiros positivos caiu para $45.9\%$.
- Mulheres brancas (**Female & White**, $N=3.867$) registraram taxa favorável de $7.8\%$ ($\text{Post-DI} = 0.306$).
- Homens negros (**Male & Black**, $N=732$) registraram taxa favorável de $12.7\%$ ($\text{Post-DI} = 0.507$).

**Constatação Teórica:** Uma análise que considerasse apenas o gênero concluiria que mulheres recebem menos predições favoráveis que homens. Todavia, a auditoria interseccional evidencia que mulheres negras enfrentam uma penalização composta: sua probabilidade de predição favorável é **6,5 vezes menor** que a de homens brancos e menos da metade da observada para mulheres brancas, validando a premissa de Kimberlé Crenshaw e Buolamwini & Gebru (2018) sobre a invisibilidade do viés interseccional em avaliações unidimensionais.

### 2. COMPAS: Disparidade Racial e de Gênero no Risco de Recidivismo
No contexto do COMPAS, a classe favorável $Y=1$ corresponde à ausência de reincidência criminal (classificação de baixo risco). Os dados revelam disparidades severas:
- **Male & African-American** ($N=1.007$): registrou a menor taxa de classificação favorável entre todos os grupos ($44.8\%$, $\text{Post-DI} = 0.585$), a menor taxa de verdadeiros positivos ($\text{TPR} = 58.1\%$) e o maior desvio absoluto de chances igualadas ($\text{AAOD} = 0.2629$).
- Em contrapartida, **Female & Caucasian** ($N=187$) obteve taxa favorável de $72.0\%$ ($\text{Post-DI} = 0.943$), $\text{TPR} = 81.4\%$ e $\text{AAOD} = 0.0436$.
- **Male & Caucasian** ($N=611$) atingiu taxa favorável de $64.1\%$ e $\text{TPR} = 73.8\%$.

O *Sensitivity Gap* superior a **$23.5$ pontos percentuais** entre homens afro-americanos e indivíduos caucasianos comprova que os modelos de aprendizado de máquina padrão internalizam e perpetuam os vieses estruturais do sistema penal, gerando uma taxa de falsos positivos desfavoráveis substancialmente maior contra jovens negros.

---

## 5.3.4 Comportamento sob Desbalanceamento Severo: O Caso do SINASC/DATASUS

O *dataset* **SINASC** (composto por $N = 2.474.535$ registros de nascidos vivos no Brasil) representa o desafio de aplicação da equidade algorítmica em dados reais de saúde pública de grande escala. A tarefa preditiva consiste na identificação de baixo peso ao nascer ($<2.500\text{g}$, classe minoritária $Y=0$, representando $9.44\%$ da população), contra o peso adequado ($\ge 2.500\text{g}$, classe majoritária $Y=1$, representando $90.56\%$). Os atributos sensíveis intersectionais combinam Raça/Cor da Mãe (5 categorias) e Faixa Etária Materna (3 categorias), totalizando 15 subgrupos.

Os resultados exibidos na Tabela 1 expõem um fenômeno crítico denominado **Colapso Trivial por Desbalanceamento**:
1. Ambos os modelos (*Random Forest* e *Gradient Boosting*) alcançaram acurácia nominal de **$90.57\%$**, exatamente idêntica à proporção a priori da classe majoritária.
2. A revocação da classe majoritária foi de $1.000$, enquanto a capacidade discriminativa real foi próxima ao acaso ($\text{ROC-AUC} \approx 0.532 \pm 0.002$).
3. Paradoxalmente, as métricas de equidade apresentaram valores aparentemente "perfeitos": $\text{Max AAOD} = 0.0000$, $\text{Sensitivity Gap} = 0.0000$ e $\text{Post-DI} = 1.000$ em todos os 15 subgrupos interseccionais.

**Diagnóstico Científico:** Essa aparente equidade absoluta é uma **ilusão estatística decorrente da degeneração do classificador**. Ao classificar $100\%$ dos partos como favoráveis (peso adequado), as taxas de verdadeiros positivos e falsos positivos tornam-se identicamente iguais a $1.0$ para todos os subgrupos de mães brancas, pretas, pardas, indígenas e amarelas em todas as faixas etárias. Como consequência:

$$|\text{TPR}_g - \text{TPR}_{\text{ref}}| = |1.0 - 1.0| = 0 \quad \text{e} \quad |\text{FPR}_g - \text{FPR}_{\text{ref}}| = |1.0 - 1.0| = 0 \implies \text{AAOD} = 0$$

Esse resultado possui implicações metodológicas de primeira ordem para a tese de doutorado:
- Demonstra a ineficácia das abordagens tradicionais de *machine learning* quando aplicadas a problemas epidemiológicos com desbalanceamento severo sem intervenções de balanceamento sensíveis à equidade.
- Evidencia que métricas de paridade estatística ou de probabilidades igualadas podem ser artificialmente satisfeitas por modelos inúteis do ponto de vista clínico.
- Justifica de forma contundente a necessidade de métodos de otimização multiobjetivo que forcem explicitamente a exploração da fronteira de compromisso entre a identificação de eventos adversos raros ($\text{PR-AUC}$, $\text{Recall}$ da classe minoritária) e a mitigação da disparidade interseccional.

---

## 5.3.5 Síntese Crítica e Justificativa para a Arquitetura Proposta

A síntese dos experimentos preliminares consolida três conclusões fundamentais que fundamentam diretamente a proposta metodológica desta qualificação de doutorado:

```
+---------------------------------------------------------------------------------------------------------+
|                                    SÍNTESE DOS RESULTADOS EXPERIMENTAIS                                  |
+------------------------------------+--------------------------------------------------------------------+
| 1. Falha da Equidade Marginal      | Subgrupos como mulheres negras (Adult) e homens negros (COMPAS)    |
|    (Univariada)                    | sofrem penalizações severas que passam despercebidas quando se     |
|                                    | avalia apenas raça ou gênero de forma isolada.                     |
+------------------------------------+--------------------------------------------------------------------+
| 2. Sensibilidade à Função          | Otimizar apenas métricas clássicas de desempenho (como acurácia    |
|    Objetivo Interna                | ou precisão) pode ampliar o Sensitivity Gap em mais de 100%.       |
+------------------------------------+--------------------------------------------------------------------+
| 3. Ilusão de Equidade sob          | Bases reais desbalanceadas (SINASC) provocam colapso de predição,  |
|    Desbalanceamento Severo         | gerando métricas de fairness artificialmente nulas que mascaram a  |
|                                    | omissão de atendimento clínico a recém-nascidos vulneráveis.       |
+------------------------------------+--------------------------------------------------------------------+
```

### Justificativa da Arquitetura Metodológica Proposta

Os resultados empíricos obtidos estabelecem a ponte conceitual necessária para justificar cada componente da metodologia proposta para o restante do programa de doutorado:

1. **Rede Neural Multitarefa com Camada Reversa de Gradiente (GRL):**
   Para mitigar o viés interseccional sem depender de suposições de independência entre atributos sensíveis, propõe-se uma arquitetura multitarefa onde um codificador latente compartilhado $E(X)$ minimiza a perda da tarefa primária $L_{\text{task}}$ enquanto ramos adversariais independentes tentam reconstruir os múltiplos atributos sensíveis ($S_1, S_2$). A introdução da Camada Reversa de Gradiente invertida por fatores de ponderação $\lambda_1, \lambda_2$:
   $$L_{\text{total}} = L_{\text{task}} - \left( \lambda_1 L_{\text{adv1}} + \lambda_2 L_{\text{adv2}} \right)$$
   força a representação latente a ser estatisticamente invariante aos subgrupos interseccionais, atacando a raiz do viés demonstrado nos dados do Adult e do COMPAS.

2. **Otimização Multiobjetivo Evolucionária com NSGA-II:**
   Os experimentos demonstraram que o espaço de *trade-offs* entre desempenho preditivo e equidade interseccional não é monótono e apresenta descontinuidades críticas (como o colapso verificado no SINASC). Ao invés de fixar arbitrariamente pesos de penalização ($\lambda$), o algoritmo evolucionário **NSGA-II** será empregado para buscar a **Fronteira de Pareto empírica** tridimensional:
   $$\max \left[ f_1(\lambda) = \text{PR-AUC}, \quad f_2(\lambda) = -\text{Max AAOD}, \quad f_3(\lambda) = -\text{Sensitivity Gap} \right]$$
   Essa formulação garante que o tomador de decisão (especialmente no setor de saúde pública e políticas sociais) possa selecionar modelos operacionais com garantias explícitas de não-degeneração e de mínima injustiça interseccional.

3. **Auditoria Pós-Processamento e Interpretabilidade com Valores de Shapley (SHAP):**
   A mera redução das métricas de disparidade não elucida *como* o modelo alterou sua lógica decisória. A integração de SHAP por subgrupo permitirá auditar se o aprendizado adversarial efetivamente eliminou o uso de atributos substitutos (*proxy features*) associados a vulnerabilidades socioeconômicas, conferindo transparência epistêmica à política pública preditiva.

Com essa base experimental e teórica consolidada, a proposta de qualificação dispõe de sustentação quantitativa robusta para demonstrar a viabilidade, a originalidade e a relevância científica da tese.
