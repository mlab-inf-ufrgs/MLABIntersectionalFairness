import streamlit as st

def t(key):
    """
    Função helper para buscar a string traduzida.
    """
    if "lang" not in st.session_state:
        st.session_state.lang = "PT"
    
    lang = st.session_state.lang
    return TEXTS.get(lang, TEXTS["PT"]).get(key, key)

TEXTS = {
    "PT": {
        "sidebar_lang": "Idioma / Language",
        "home_page_title": "Diagnóstico de Viés Interseccional",
        "home_title": "Análise de impacto cumulativo do viés social",
        "home_nav": "**Navegação:**\n- **Dados**: Diagnóstico pré-treinamento e Análise Exploratória. Avalia o viés em nível de dados.\n- **Modelos**: Avaliação de disparidades e trade-offs de justiça algorítmica pós-treinamento.\n\nSelecione a aba desejada no menu lateral.",
        "models_page_title": "Modelos",
        "models_title": "Avaliação Pós-Treinamento",
        "models_info": "",

        # Proveniência
        "provenance_header": "🔬 Metodologia e Proveniência dos Experimentos",
        "provenance_desc": (
            "Os resultados desta aba foram produzidos por um pipeline de **Nested Cross-Validation** "
            "independente do servidor Streamlit, garantindo que nenhum modelo seja treinado durante a navegação.\n\n"
            "**Pipeline:** Outer CV K=3 (avaliação) + Inner CV K=3 (tuning via `RandomizedSearchCV`, `n_iter=30`).\n"
            "**Modelos:** Random Forest, Gradient Boosting (scikit-learn).\n"
            "**Métricas de otimização avaliadas:** Accuracy, Recall, Precision, AUROC, AUPRC.\n"
            "**Atributos sensíveis excluídos das features** (modelo fairness-unaware, conforme artigo de referência).\n\n"
            "Grupo de referência para DI/AAOD: subgrupo interseccional com **maior taxa de predição favorável** "
            "na fold de teste (determinado dinamicamente por fold).\n"
            "**Max AAOD** = AAOD do subgrupo mais prejudicado (estimativa conservadora, protege o pior caso)."
        ),
        "provenance_run_ts": "Executado em",
        "provenance_outer_k": "Outer K",
        "provenance_inner_k": "Inner K / Busca",
        "provenance_dryrun_warn": "⚠️ Estes são resultados de **dry-run** (1 fold, 5 iterações). Execute o script completo para resultados definitivos.",
        "no_results_warn": "Nenhum resultado encontrado. Execute o script de experimentos primeiro:",
        "load_error": "Erro ao carregar os resultados:",

        # Bloco 1
        "b1_models_header": "1. Seleção de Contexto e Métricas Globais",
        "select_attrs_combo": "Combinação de Atributos Interseccionais:",
        "select_model": "Modelo:",
        "select_opt_metric": "Métrica de Otimização:",
        "global_metrics_header": "Desempenho Global (média ± desvio padrão — outer folds)",
        "max_aaod_metric": "Max AAOD Interseccional",
        "sens_gap_metric": "Sensitivity Gap",
        "fairness_def_expander": "ℹ️ Definição das métricas de fairness",
        "fairness_def_text": (
            "* **Max AAOD (Average Absolute Odds Difference):** "
            "`0.5 × (|FPR_u − FPR_ref| + |TPR_u − TPR_ref|)` calculado para cada subgrupo `u` em relação ao "
            "grupo de referência (maior taxa favorável). Reportamos o **máximo** entre os subgrupos para "
            "adotar postura conservadora e proteger o pior caso.\n"
            "* **Sensitivity Gap:** `max(TPR) − min(TPR)` entre subgrupos viáveis (N ≥ 30). "
            "Mede a disparidade máxima na capacidade do modelo de identificar corretamente o desfecho favorável.\n"
            "* **Post-training DI:** `P(ŷ=favorável | grupo=u) / P(ŷ=favorável | grupo=referência)`. "
            "Valores < 0.8 ou > 1.25 indicam disparidade pela Regra dos 80%."
        ),

        # Bloco 2 — Pareto
        "pareto_header": "2. Trade-off: Desempenho × Injustiça Interseccional (Pareto)",
        "pareto_desc": (
            "Cada ponto representa uma combinação **(modelo × métrica de otimização)**. "
            "O **quadrante ideal** é o canto inferior direito: alta performance e baixa injustiça. "
            "Use este gráfico para identificar qual estratégia de otimização melhor equilibra eficiência e equidade."
        ),
        "pareto_xaxis_label": "Métrica de performance (eixo X):",
        "pareto_ideal_label": "← Região ideal",
        "max_aaod_label": "Max AAOD Interseccional",
        "model_legend": "Modelo",
        "opt_metric_legend": "Métrica de Otimização",
        "pareto_caption": (
            "Barras de erro representam o desvio padrão entre as outer folds. "
            "Pontos mais à direita e mais abaixo indicam melhor trade-off performance/fairness."
        ),

        # Bloco 3 — Dumbbell
        "dumbbell_header": "3. Disparate Impact por Subgrupo Interseccional: Pré vs. Pós-Treinamento",
        "dumbbell_desc": (
            "Para cada subgrupo, o **Pre-DI** (calculado sobre os rótulos da base completa) e o **Post-DI** "
            "(média entre as outer folds, calculado sobre as predições do modelo selecionado). Em ambos, a "
            "referência é o subgrupo com maior taxa favorável. "
            "A linha cinza em DI = 1,0 indica paridade; as linhas laranjas em 0,8 e 1,25 delimitam a Regra dos 80%."
        ),
        "post_di_label": "DI Pós-treinamento",
        "pre_di_label": "DI Pré-treinamento",
        "moment_pre": "Pré-treino (dados)",
        "moment_post": "Pós-treino (predições)",
        "data_rate_label": "Taxa nos dados",
        "pred_rate_label": "Taxa de predição",
        "dumbbell_caption": (
            "O segmento liga o DI dos dados ao DI das predições. Quando o ponto laranja fica à esquerda do azul, "
            "o modelo ampliou a disparidade do subgrupo. Barra de erro = ±1 desvio-padrão do Post-DI entre outer folds."
        ),
        "dumbbell_table_expander": "📋 Tabela de dados completa por subgrupo",
        "latex_subgroup_expander": "📄 Código LaTeX desta tabela por subgrupo",

        # Comparação entre modelos
        "comparison_header": "Comparação entre Todos os Modelos e Configurações",
        "comparison_desc": (
            "Desempenho preditivo e equidade interseccional de cada modelo testado neste dataset: Random Forest e "
            "Gradient Boosting em cada métrica de otimização da busca de hiperparâmetros, e a MLP em cada valor de λ."
        ),
        "comparison_caption": "Média ± desvio-padrão entre as 3 outer folds. ↑: maior é melhor; ↓: menor é melhor.",
        "latex_tables_expander": "📄 Código LaTeX desta tabela",
        "latex_tables_desc": (
            "Formato do texto da tese: vírgula decimal e linhas \\hline. Requer os pacotes `multirow` e `pdflscape` "
            "e a opção de posicionamento `[H]` (pacote `float`)."
        ),
        "btn_download_tex": "📥 Baixar .tex",
        "no_subgroup_data": "Dados de subgrupos não disponíveis para esta combinação.",

        # Bloco 4 — Ranking
        "ranking_header": "4. Ranking: Qual Métrica de Otimização É Mais Justa?",
        "ranking_desc": (
            "Heatmap do **AAOD médio** (entre outer folds) por combinação "
            "*(métrica de otimização × subgrupo interseccional)*. "
            "Cores mais escuras indicam maior injustiça para aquele subgrupo sob aquela estratégia de otimização. "
            "Identifica qual escolha de função objetivo é mais prejudicial a grupos específicos."
        ),
        "ranking_model_select": "Modelo para o ranking:",
        "ranking_caption": (
            "Leitura: cada célula mostra o AAOD médio daquele subgrupo quando o modelo é otimizado "
            "pela métrica da linha. Um AAOD próximo de zero indica paridade de erros com o grupo de referência."
        ),
        "export_ranking_csv": "📥 Exportar Ranking (CSV)",

        # Visão Consolidada Multi-Dataset
        "tab_consolidated_title": "🌐 Visão Consolidada (Cross-Dataset)",
        "tab_detailed_title": "🔍 Auditoria por Dataset",
        "consolidated_header": "Desempenho Global e Disparidades Transversais",
        "consolidated_desc": "Comparação transversal dos modelos avaliados em múltiplos domínios (crédito, justiça e saúde).",
        "fig1_interactive_title": "Distribuição de Desempenho e Disparidade por Modelo",
        "fig1_select_left": "Métrica do Painel Esquerdo",
        "fig1_select_right": "Métrica do Painel Direito",
        "btn_download_highres": "📥 Baixar Figura em Alta Resolução (PNG)",
        "best_in_class_header": "🏆 Modelos de Destaque por Dataset",
        "summary_table_header": "📊 Tabela Comparativa Consolidada",
        "latex_code_header": "📄 Código LaTeX Pronto para Artigo / Tese",
        "copy_latex_desc": "Copie o código abaixo diretamente para o seu documento LaTeX (usa o pacote booktabs):",
        
        # Dados.py Texts
        "data_page_title": "Dados (EDA)",
        "data_title": "Análise Exploratória e Viés Social Pré-Treino",
        "select_country": "Selecione o país/região:",
        "select_dataset": "Selecione o Dataset:",
        "select_state": "Selecione o Estado (UF):",
        "loading": "Carregando e processando dados...",
        
        "domain_label": "**Domínio:**",
        "protected_label": "**Atributos protegidos (Demográficos):**",
        "proxies_label": "**Proxies (Socioeconômicos/Comportamentais):**",
        "original_source": "🔗 [Fonte original]",
        
        "n_original": "N Original (Bruto)",
        "n_processed": "N Pós-processamento",
        "year": "Ano",
        "class_dist": "Dist. de Classes (Alvo)",
        "class_dist_help": "Proporção global entre a classe majoritária e minoritária da variável-alvo. Mede o quão desbalanceados estão os desfechos em toda a base de dados.",
        
        # Bloco 1
        "b1_header": "1. Distribuição Geral do Alvo",
        "target": "**Alvo:**",
        "favorable_class": "**Classe favorável:**",
        "filter_classes": "Filtrar classes:",
        "class_label": "Classe",
        "freq_label": "Frequência",
        
        # Bloco 2
        "b2_header": "2. Viés Unidimensional",
        "b2_desc": "Distribuição dos resultados favoráveis por subgrupo demográfico",
        "view": "Visão",
        "view_by_attr": "Visão por atributo",
        "view_agg": "Visão agregada",
        "subgroup": "Subgrupo",
        "favorable_rate": "Taxa Favorável",
        "global_mean": "Média Global",
        "global_mean_legend": "A linha pontilhada vermelha representa a Média Global",
        "select_attr": "Selecione o atributo para inspecionar:",
        
        "fairness_metrics_for": "**Métricas de Equidade para",
        "groups_identified": "Grupos identificados automaticamente para cálculo:<br>**Privilegiado** = `{}` (maior taxa de sucesso) | **Desprivilegiado** = `{}` (menor taxa de sucesso).",
        "ci_metric": "Desbalanceamento de Classe (CI)",
        "di_metric": "DI Pré-treino",
        "kl_metric": "Divergência KL",
        "ks_metric": "Estatística KS",
        "di_adv_metric": "DI Adverso",
        "di_comparison_title": "DI Marginal vs. Interseccional (mesma escala)",
        "di_comparison_desc": "Para cada atributo isolado e cada combinação de atributos, compara o pior e o melhor subgrupo viável (N ≥ 100). O **DI (Favorável)** é a razão entre as taxas favoráveis (pior / melhor); o **DI (Adverso)** é a razão entre as taxas de desfecho desfavorável dos mesmos subgrupos (pior / melhor).",
        "di_comparison_caption": "Destaques: DI (Favorável) < 0,80 ou DI (Adverso) > 1,25 (Regra dos 80%). Em bases com taxa favorável próxima de 1, o DI favorável tende a 1 por construção, enquanto o DI adverso preserva a disparidade relativa. '∞' indica que o melhor subgrupo não possui nenhum desfecho adverso.",
        "di_comparison_empty": "Não há subgrupos viáveis (N ≥ 100) suficientes para a comparação.",
        "understand_metrics": "ℹ️ Entenda as Métricas",
        "metrics_explanation": "* **Desbalanceamento de Classe (CI)**: Mede o desbalanceamento demográfico entre os dois grupos extremos listados acima. Varia de -1 (todas as amostras no grupo desprivilegiado) a 1 (todas no privilegiado). Ideal: 0.\n* **DI Pré-treino (Disparate Impact)**: Razão entre a chance do grupo desprivilegiado pertencer à classe alvo favorável e a do grupo privilegiado. Valores < 0.8 ou > 1.25 indicam disparidade (Regra dos 80%).\n* **DI Adverso**: Razão entre as taxas de desfecho desfavorável do grupo desprivilegiado e do privilegiado (mesmos grupos do DI). Valores > 1.25 indicam disparidade. Útil em bases desbalanceadas, em que o DI favorável tende a 1 por construção.\n* **Divergência KL**: Mede a divergência (distância) entre as distribuições de probabilidade de resultados do grupo privilegiado e desprivilegiado.\n* **Estatística KS (Kolmogorov-Smirnov)**: Mede a distância máxima entre as distribuições acumuladas dos dois grupos. Valores altos indicam desigualdade na distribuição.",
        
        # Bloco 3
        "b3_header": "3. Viés Interseccional",
        "b3_desc": "Selecione múltiplos atributos para análise de interseccionalidade e Justice Gerrymandering.",
        "select_multi_attr": "Selecione 2 ou mais atributos:",
        "warn_multi_attr": "Selecione pelo menos 2 atributos para análise interseccional.",
        "tab_subgroups": "Auditoria de Subgrupos",
        "tab_pairs": "Auditoria em Pares (Global)",
        
        "viz_intersec_2": "Visualização Interseccional (2 Atributos)",
        "caption_translucent": "Barras translúcidas indicam N < 100 (subgrupo inviável estatisticamente). Linha vermelha = Média Global.",
        
        "audit_gerry_3": "Auditoria de Justice Gerrymandering (3+ Atributos)",
        "calc_gerry": "Calculando métricas de Gerrymandering...",
        "audit_gerry_3_desc": "Cada linha é um subgrupo interseccional. **Gap Real** = taxa favorável do subgrupo − taxa global. **Gap Esperado** = gap da identidade marginal mais desfavorecida do subgrupo (ex.: para Mulher & Negro & Jovem, o menor entre os gaps de Mulher, Negro e Jovem). **Viés Oculto** = Gap Real − Gap Esperado: valores negativos indicam que o subgrupo está pior do que sua identidade mais desfavorecida, isolada, sugeriria.",
        "caption_inviable": "Valores em pontos percentuais (p.p.). Vereditos: **Alto Viés Oculto** = Viés Oculto < −0,05 p.p.; **Alto Viés Direto** = Gap Real < −5 p.p., sem viés oculto; **Inviável** = N < 100 (estimativa instável). O limiar de 0,05 p.p. é descritivo, não um teste de significância estatística.",
        "export_csv": "📥 Exportar Dados (CSV Longo Consolidado)",
        
        "cddl_header": "Disparidade Demográfica Condicional (CDDL)",
        "no_proxy": "Nenhum proxy definido",
        "calc_cddl": "Calculando CDDL...",
        "caption_cddl": "Valores positivos indicam que a classe desprivilegiada sofre mais resultados desfavoráveis (e menos favoráveis) condicionado aos estratos do proxy.",
        "warn_metadata": "Metadados incompletos para definir grupo privilegiado/desprivilegiado principal.",
        
        "audit_pairs": "Auditoria de Gerrymandering em Pares (Global)",
        "audit_pairs_desc": "Esta visão varre **todos os pares** de atributos selecionados. Aqui o gap é a **amplitude** (maior − menor taxa favorável): **Gap Esperado** é a maior amplitude entre os dois atributos isolados e **Gap Real** é a amplitude entre os subgrupos do par (N ≥ 100). **Viés Oculto** positivo indica que a interseção revela uma disparidade maior do que qualquer um dos atributos isolados. Por usar amplitude, e não distância à taxa global, os valores não são diretamente comparáveis aos da aba de subgrupos.",
        "sort_note_pairs": "Pares ordenados pelo Gap Real (Interseccional), do maior para o menor, ou seja, pela disparidade observada na interseção, independentemente dos atributos isolados.",
        "caption_pairs": "Valores em pontos percentuais (p.p.). Destaque: Viés Oculto > 0,05 p.p. Trata-se de um limiar descritivo, não de um teste de significância estatística: diferenças pequenas podem refletir ruído amostral, sobretudo em subgrupos com N próximo de 100.",
        "scanning_pairs": "Varrendo pares...",
        "gap_audit_title": "Auditoria de Gap: Disparidade Marginal vs. Interseccional",
        
        # Bloco 4
        "b4_header": "4. Matriz de Correlação (Risco de Proxies)",
        "b4_desc": "O **V de Cramér** mede a associação estatística entre variáveis categóricas (0 = sem associação, 1 = associação perfeita). Valores altos entre um *proxy* socioeconômico e um atributo protegido indicam alto risco de que modelos descubram a classe sensível indiretamente (*Redlining*).",
        "calc_cramer": "Calculando V de Cramér...",
        "cramer_v": "V de Cramér",
        "insufficient_attrs": "Atributos insuficientes para gerar a matriz de correlação.",
        
        # Table Headers Translated (Used dynamically or directly)
        "tbl_subgroup": "Subgrupo",
        "tbl_fav_rate": "Taxa Favorável",
        "tbl_real_gap": "Gap Real (Interseccional)",
        "tbl_exp_gap": "Gap Esperado (Marginal Máx)",
        "tbl_exp_gap_worst": "Gap Esperado (Pior Marginal)",
        "tbl_hidden_bias": "Viés Oculto (Excedente)",
        "tbl_prio_score": "Score de Prioridade",
        "tbl_pre_di": "DI Pré-treinamento",
        "tbl_audit_verdict": "Veredito da Auditoria",
        "tbl_metric": "Métrica",
        "tbl_value": "Valor",
        "tbl_pair": "Par de Intersecção",
        "tbl_gap_a": "Gap Indiv. A",
        "tbl_gap_b": "Gap Indiv. B",
        "tbl_gap_type": "Tipo de Gap",
        "tbl_gap_amplitude": "Amplitude do Gap (p.p.)"
    },
    "EN": {
        "sidebar_lang": "Idioma / Language",
        "home_page_title": "Intersectional Bias Diagnosis",
        "home_title": "Cumulative impact analysis of social bias",
        "home_nav": "**Navigation:**\n- **Data**: Pre-training diagnosis and Exploratory Data Analysis (EDA). Evaluates inherent data bias and Gerrymandering audit.\n- **Models**: Post-training evaluation of disparities and algorithmic fairness trade-offs.\n\nSelect the desired tab in the sidebar.",
        "models_page_title": "Models",
        "models_title": "Post-Training Evaluation — Intersectional Fairness",
        "models_info": "",

        # Provenance
        "provenance_header": "🔬 Methodology and Experiment Provenance",
        "provenance_desc": (
            "The results on this tab were produced by a **Nested Cross-Validation** pipeline "
            "that runs independently of the Streamlit server, ensuring no model is trained during browsing.\n\n"
            "**Pipeline:** Outer CV K=3 (evaluation) + Inner CV K=3 (tuning via `RandomizedSearchCV`, `n_iter=30`).\n"
            "**Models:** Random Forest, Gradient Boosting (scikit-learn).\n"
            "**Optimization metrics evaluated:** Accuracy, Recall, Precision, AUROC, AUPRC.\n"
            "**Sensitive attributes excluded from features** (fairness-unaware model, as per reference paper).\n\n"
            "Reference group for DI/AAOD: intersectional subgroup with the **highest favorable prediction rate** "
            "in the test fold (determined dynamically per fold).\n"
            "**Max AAOD** = AAOD of the most disadvantaged subgroup (conservative estimate, protects the worst case)."
        ),
        "provenance_run_ts": "Run at",
        "provenance_outer_k": "Outer K",
        "provenance_inner_k": "Inner K / Search",
        "provenance_dryrun_warn": "⚠️ These are **dry-run** results (1 fold, 5 iterations). Run the full script for definitive results.",
        "no_results_warn": "No results found. Run the experiment script first:",
        "load_error": "Error loading results:",

        # Block 1
        "b1_models_header": "1. Context Selection and Global Metrics",
        "select_attrs_combo": "Intersectional Attribute Combination:",
        "select_model": "Model:",
        "select_opt_metric": "Optimization Metric:",
        "global_metrics_header": "Global Performance (mean ± std — outer folds)",
        "max_aaod_metric": "Max Intersectional AAOD",
        "sens_gap_metric": "Sensitivity Gap",
        "fairness_def_expander": "ℹ️ Fairness metrics definitions",
        "fairness_def_text": (
            "* **Max AAOD (Average Absolute Odds Difference):** "
            "`0.5 × (|FPR_u − FPR_ref| + |TPR_u − TPR_ref|)` computed for each subgroup `u` relative to the "
            "reference group (highest favorable rate). We report the **maximum** across subgroups to "
            "adopt a conservative stance and protect the worst case.\n"
            "* **Sensitivity Gap:** `max(TPR) − min(TPR)` across viable subgroups (N ≥ 30). "
            "Measures the maximum disparity in the model's ability to correctly identify the favorable outcome.\n"
            "* **Post-training DI:** `P(ŷ=favorable | group=u) / P(ŷ=favorable | group=reference)`. "
            "Values < 0.8 or > 1.25 indicate disparity under the 80% Rule."
        ),

        # Block 2 — Pareto
        "pareto_header": "2. Trade-off: Performance × Intersectional Unfairness (Pareto)",
        "pareto_desc": (
            "Each point represents a **(model × optimization metric)** combination. "
            "The **ideal quadrant** is the lower right: high performance and low unfairness. "
            "Use this chart to identify which optimization strategy best balances efficiency and equity."
        ),
        "pareto_xaxis_label": "Performance metric (X axis):",
        "pareto_ideal_label": "← Ideal region",
        "max_aaod_label": "Max Intersectional AAOD",
        "model_legend": "Model",
        "opt_metric_legend": "Optimization Metric",
        "pareto_caption": (
            "Error bars represent standard deviation across outer folds. "
            "Points further right and lower indicate a better performance/fairness trade-off."
        ),

        # Block 3 — Dumbbell
        "dumbbell_header": "3. Disparate Impact by Intersectional Subgroup: Pre- vs. Post-Training",
        "dumbbell_desc": (
            "For each subgroup, the **Pre-DI** (computed on the full dataset labels) and the **Post-DI** "
            "(mean across outer folds, computed on the selected model's predictions). In both, the reference is "
            "the subgroup with the highest favorable rate. "
            "The gray line at DI = 1.0 indicates parity; orange lines at 0.8 and 1.25 delimit the 80% Rule."
        ),
        "post_di_label": "Post-training DI",
        "pre_di_label": "Pre-training DI",
        "moment_pre": "Pre-training (data)",
        "moment_post": "Post-training (predictions)",
        "data_rate_label": "Rate in data",
        "pred_rate_label": "Prediction rate",
        "dumbbell_caption": (
            "The segment links the data DI to the prediction DI. When the orange point lies left of the blue one, "
            "the model widened the subgroup's disparity. Error bar = ±1 std of Post-DI across outer folds."
        ),
        "dumbbell_table_expander": "📋 Full data table by subgroup",
        "latex_subgroup_expander": "📄 LaTeX code for this subgroup table",

        # Model comparison
        "comparison_header": "Comparison Across All Models and Configurations",
        "comparison_desc": (
            "Predictive performance and intersectional fairness of every model tested on this dataset: Random Forest "
            "and Gradient Boosting for each hyperparameter-search optimization metric, and MLP for each λ value."
        ),
        "comparison_caption": "Mean ± std across the 3 outer folds. ↑: higher is better; ↓: lower is better.",
        "latex_tables_expander": "📄 LaTeX code for this table",
        "latex_tables_desc": (
            "Thesis text format: decimal comma and \\hline rules. Requires the `multirow` and `pdflscape` packages "
            "and the `[H]` placement option (`float` package)."
        ),
        "btn_download_tex": "📥 Download .tex",
        "no_subgroup_data": "Subgroup data not available for this combination.",

        # Block 4 — Ranking
        "ranking_header": "4. Ranking: Which Optimization Metric Is Fairest?",
        "ranking_desc": (
            "Heatmap of **mean AAOD** (across outer folds) per "
            "*(optimization metric × intersectional subgroup)* combination. "
            "Darker colors indicate greater unfairness for that subgroup under that optimization strategy. "
            "Identifies which objective function choice is most harmful to specific groups."
        ),
        "ranking_model_select": "Model for ranking:",
        "ranking_caption": (
            "Reading: each cell shows the mean AAOD of that subgroup when the model is optimized "
            "by the row metric. An AAOD near zero indicates parity of errors with the reference group."
        ),
        "export_ranking_csv": "📥 Export Ranking (CSV)",

        # Consolidated Multi-Dataset View
        "tab_consolidated_title": "🌐 Consolidated View (Cross-Dataset)",
        "tab_detailed_title": "🔍 Audit by Dataset",
        "consolidated_header": "Global Performance and Cross-Dataset Disparities",
        "consolidated_desc": "Cross-domain comparison of models evaluated across credit, criminal justice, and public health.",
        "fig1_interactive_title": "Performance and Disparity Distribution by Model",
        "fig1_select_left": "Left Panel Metric",
        "fig1_select_right": "Right Panel Metric",
        "btn_download_highres": "📥 Download High-Resolution Figure (PNG)",
        "best_in_class_header": "🏆 Best-in-Class Models by Dataset",
        "summary_table_header": "📊 Consolidated Comparison Table",
        "latex_code_header": "📄 Ready-to-use LaTeX Code for Paper / Thesis",
        "copy_latex_desc": "Copy the code below directly into your LaTeX document (uses the booktabs package):",
        
        # Dados.py Texts
        "data_page_title": "Data (EDA)",
        "data_title": "Exploratory Analysis and Pre-Training Social Bias",
        "select_country": "Select country/region:",
        "select_dataset": "Select Dataset:",
        "select_state": "Select State (UF):",
        "loading": "Loading and processing data...",
        
        "domain_label": "**Domain:**",
        "protected_label": "**Protected Attributes (Demographic):**",
        "proxies_label": "**Proxies (Socioeconomic/Behavioral):**",
        "original_source": "🔗 [Original Source]",
        
        "n_original": "Original N (Raw)",
        "n_processed": "Post-processing N",
        "year": "Year",
        "class_dist": "Class Distribution (Target)",
        "class_dist_help": "Global proportion between the majority and minority class of the target variable. Measures how imbalanced the outcomes are in the entire dataset.",
        
        # Bloco 1
        "b1_header": "1. General Target Distribution",
        "target": "**Target:**",
        "favorable_class": "**Favorable class:**",
        "filter_classes": "Filter classes:",
        "class_label": "Class",
        "freq_label": "Frequency",
        
        # Bloco 2
        "b2_header": "2. Unidimensional Bias",
        "b2_desc": "Distribution of favorable results by demographic subgroup",
        "view": "View",
        "view_by_attr": "View by attribute",
        "view_agg": "Aggregated view",
        "subgroup": "Subgroup",
        "favorable_rate": "Favorable Rate",
        "global_mean": "Global Mean",
        "global_mean_legend": "The red dashed line represents the Global Mean",
        "select_attr": "Select the attribute to inspect:",
        
        "fairness_metrics_for": "**Fairness Metrics for",
        "groups_identified": "Groups automatically identified for calculation:<br>**Privileged** = `{}` (highest success rate) | **Unprivileged** = `{}` (lowest success rate).",
        "ci_metric": "Class Imbalance (CI)",
        "di_metric": "Pre-training DI",
        "kl_metric": "KL Divergence",
        "ks_metric": "KS Statistic",
        "di_adv_metric": "Adverse DI",
        "di_comparison_title": "Marginal vs. Intersectional DI (same scale)",
        "di_comparison_desc": "For each single attribute and each combination of attributes, compares the worst and best viable subgroups (N ≥ 100). **DI (Favorável)** is the ratio of favorable rates (worst / best); **DI (Adverso)** is the ratio of unfavorable outcome rates for the same subgroups (worst / best).",
        "di_comparison_caption": "Highlights: DI (Favorável) < 0.80 or DI (Adverso) > 1.25 (80% Rule). When the favorable rate is close to 1, favorable DI tends to 1 by construction, while adverse DI preserves the relative disparity. '∞' means the best subgroup has no adverse outcomes.",
        "di_comparison_empty": "Not enough viable subgroups (N ≥ 100) for the comparison.",
        "understand_metrics": "ℹ️ Understand the Metrics",
        "metrics_explanation": "* **Class Imbalance (CI)**: Measures demographic imbalance between the two extreme groups listed above. Varies from -1 (all samples in the unprivileged group) to 1 (all in the privileged group). Ideal: 0.\n* **Pre-training DI (Disparate Impact)**: Ratio between the chance of the unprivileged group belonging to the favorable target class and that of the privileged group. Values < 0.8 or > 1.25 indicate disparity (80% Rule).\n* **Adverse DI**: Ratio between the unfavorable outcome rates of the unprivileged and privileged groups (same groups as DI). Values > 1.25 indicate disparity. Useful for imbalanced datasets, where favorable DI tends to 1 by construction.\n* **KL Divergence**: Measures the divergence (distance) between the probability distributions of outcomes for the privileged and unprivileged groups.\n* **KS Statistic (Kolmogorov-Smirnov)**: Measures the maximum distance between the cumulative distributions of the two groups. High values indicate inequality in the distribution.",
        
        # Bloco 3
        "b3_header": "3. Intersectional Bias",
        "b3_desc": "Select multiple attributes for intersectionality and Justice Gerrymandering analysis.",
        "select_multi_attr": "Select 2 or more attributes:",
        "warn_multi_attr": "Select at least 2 attributes for intersectional analysis.",
        "tab_subgroups": "Subgroup Audit",
        "tab_pairs": "Pairwise Audit (Global)",
        
        "viz_intersec_2": "Intersectional Visualization (2 Attributes)",
        "caption_translucent": "Translucent bars indicate N < 100 (statistically unviable subgroup). Red line = Global Mean.",
        
        "audit_gerry_3": "Justice Gerrymandering Audit (3+ Attributes)",
        "calc_gerry": "Calculating Gerrymandering metrics...",
        "audit_gerry_3_desc": "Each row is an intersectional subgroup. **Real Gap** = subgroup favorable rate − global rate. **Expected Gap** = gap of the subgroup's most disadvantaged marginal identity (e.g., for Woman & Black & Young, the lowest among the gaps of Woman, Black and Young). **Hidden Bias** = Real Gap − Expected Gap: negative values mean the subgroup is worse off than its most disadvantaged identity alone would suggest.",
        "caption_inviable": "Values in percentage points (p.p.). Verdicts: **High Hidden Bias** = Hidden Bias < −0.05 p.p.; **High Direct Bias** = Real Gap < −5 p.p., without hidden bias; **Inviable** = N < 100 (unstable estimate). The 0.05 p.p. threshold is descriptive, not a statistical significance test.",
        "export_csv": "📥 Export Data (Long Consolidated CSV)",
        
        "cddl_header": "Conditional Demographic Disparity in Labels (CDDL)",
        "no_proxy": "No proxy defined",
        "calc_cddl": "Calculating CDDL...",
        "caption_cddl": "Positive values indicate that the unprivileged class suffers more unfavorable results (and fewer favorable) conditioned to proxy strata.",
        "warn_metadata": "Incomplete metadata to define primary privileged/unprivileged group.",
        
        "audit_pairs": "Pairwise Gerrymandering Audit (Global)",
        "audit_pairs_desc": "This view scans **all pairs** of selected attributes. Here the gap is the **range** (highest − lowest favorable rate): **Expected Gap** is the largest range between the two attributes taken alone and **Real Gap** is the range across the pair's subgroups (N ≥ 100). Positive **Hidden Bias** means the intersection reveals a larger disparity than either attribute alone. Because it uses ranges rather than distance to the global rate, values are not directly comparable to the subgroup tab.",
        "sort_note_pairs": "Pairs sorted by Real Gap (Intersectional), from largest to smallest, i.e., by the disparity observed at the intersection, regardless of the single attributes.",
        "caption_pairs": "Values in percentage points (p.p.). Highlight: Hidden Bias > 0.05 p.p. This is a descriptive threshold, not a statistical significance test: small differences may reflect sampling noise, especially for subgroups with N close to 100.",
        "scanning_pairs": "Scanning pairs...",
        "gap_audit_title": "Gap Audit: Marginal vs. Intersectional Disparity",
        
        # Bloco 4
        "b4_header": "4. Correlation Matrix (Proxy Risk)",
        "b4_desc": "**Cramér's V** measures the statistical association between categorical variables (0 = no association, 1 = perfect association). High values between a socioeconomic *proxy* and a protected attribute indicate a high risk that models will discover the sensitive class indirectly (*Redlining*).",
        "calc_cramer": "Calculating Cramér's V...",
        "cramer_v": "Cramér's V",
        "insufficient_attrs": "Insufficient attributes to generate correlation matrix.",
        
        # Table Headers Translated (Used dynamically or directly)
        "tbl_subgroup": "Subgroup",
        "tbl_fav_rate": "Favorable Rate",
        "tbl_real_gap": "Real Gap (Intersectional)",
        "tbl_exp_gap": "Expected Gap (Max Marginal)",
        "tbl_exp_gap_worst": "Expected Gap (Worst Marginal)",
        "tbl_hidden_bias": "Hidden Bias (Surplus)",
        "tbl_prio_score": "Priority Score",
        "tbl_pre_di": "Pre-training DI",
        "tbl_audit_verdict": "Audit Verdict",
        "tbl_metric": "Metric",
        "tbl_value": "Value",
        "tbl_pair": "Intersection Pair",
        "tbl_gap_a": "Indiv. Gap A",
        "tbl_gap_b": "Indiv. Gap B",
        "tbl_gap_type": "Gap Type",
        "tbl_gap_amplitude": "Gap Amplitude (p.p.)"
    }
}
