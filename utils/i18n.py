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
        "home_nav": "**Navegação:**\n- **Dados**: Diagnóstico pré-treinamento e Análise Exploratória (EDA). Avalia o viés inerente aos dados e auditoria de Gerrymandering.\n- **Modelos**: (Em Breve) Avaliação de disparidades e trade-offs de justiça algorítmica pós-treinamento.\n\nSelecione a aba desejada no menu lateral.",
        "models_page_title": "Modelos",
        "models_title": "Avaliação Pós-Treinamento — Fairness Interseccional",
        "models_info": "",

        # Proveniência
        "provenance_header": "🔬 Metodologia e Proveniência dos Experimentos",
        "provenance_desc": (
            "Os resultados desta aba foram produzidos por um pipeline de **Nested Cross-Validation** "
            "independente do servidor Streamlit, garantindo que nenhum modelo seja treinado durante a navegação.\n\n"
            "**Pipeline:** Outer CV K=3 (avaliação) + Inner CV K=3 (tuning via `RandomizedSearchCV`, `n_iter=30`).\n"
            "**Modelos:** Random Forest, Gradient Boosting (scikit-learn).\n"
            "**Métricas de otimização avaliadas:** Accuracy, Recall, Precision, ROC-AUC, PR-AUC.\n"
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
        "dumbbell_header": "3. Disparate Impact Pós-Treinamento por Subgrupo Interseccional",
        "dumbbell_desc": (
            "Média do **Post-training DI** (± desvio padrão entre folds) para cada subgrupo interseccional. "
            "A linha cinza em DI=1.0 indica paridade perfeita. "
            "Linhas laranjas em 0.8 e 1.25 delimitam a Regra dos 80% (zona de disparidade)."
        ),
        "post_di_label": "DI Pós-treinamento",
        "dumbbell_caption": (
            "🔴 DI < 0.8 (subgrupo desfavorecido) | 🟠 DI > 1.25 (superrepresentado) | 🔵 dentro da faixa aceitável. "
            "Barras de erro = ±1 desvio padrão entre outer folds."
        ),
        "dumbbell_table_expander": "📋 Tabela de dados completa por subgrupo",
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
        "understand_metrics": "ℹ️ Entenda as Métricas",
        "metrics_explanation": "* **Desbalanceamento de Classe (CI)**: Mede o desbalanceamento demográfico entre os dois grupos extremos listados acima. Varia de -1 (todas as amostras no grupo desprivilegiado) a 1 (todas no privilegiado). Ideal: 0.\n* **DI Pré-treino (Disparate Impact)**: Razão entre a chance do grupo desprivilegiado pertencer à classe alvo favorável e a do grupo privilegiado. Valores < 0.8 ou > 1.2 indicam disparidade (Regra dos 80%).\n* **Divergência KL**: Mede a divergência (distância) entre as distribuições de probabilidade de resultados do grupo privilegiado e desprivilegiado.\n* **Estatística KS (Kolmogorov-Smirnov)**: Mede a distância máxima entre as distribuições acumuladas dos dois grupos. Valores altos indicam desigualdade na distribuição.",
        
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
        "caption_inviable": "Subgrupos com N < 100 são marcados como 'Inviável'. DI pré-treinamento indica Pre-training Disparate Impact contra a média global.",
        "export_csv": "📥 Exportar Dados (CSV Longo Consolidado)",
        
        "cddl_header": "Disparidade Demográfica Condicional (CDDL)",
        "no_proxy": "Nenhum proxy definido",
        "calc_cddl": "Calculando CDDL...",
        "caption_cddl": "Valores positivos indicam que a classe desprivilegiada sofre mais resultados desfavoráveis (e menos favoráveis) condicionado aos estratos do proxy.",
        "warn_metadata": "Metadados incompletos para definir grupo privilegiado/desprivilegiado principal.",
        
        "audit_pairs": "Auditoria de Gerrymandering em Pares (Global)",
        "audit_pairs_desc": "Esta visão varre **todos os pares possíveis** entre os atributos selecionados para encontrar a disparidade máxima na margem versus na interseção.",
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
        "tbl_gap_amplitude": "Amplitude do Gap (%)"
    },
    "EN": {
        "sidebar_lang": "Idioma / Language",
        "home_page_title": "Intersectional Bias Diagnosis",
        "home_title": "Cumulative impact analysis of social bias",
        "home_nav": "**Navigation:**\n- **Data**: Pre-training diagnosis and Exploratory Data Analysis (EDA). Evaluates inherent data bias and Gerrymandering audit.\n- **Models**: (Coming Soon) Post-training evaluation of disparities and algorithmic fairness trade-offs.\n\nSelect the desired tab in the sidebar.",
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
            "**Optimization metrics evaluated:** Accuracy, Recall, Precision, ROC-AUC, PR-AUC.\n"
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
        "dumbbell_header": "3. Post-Training Disparate Impact by Intersectional Subgroup",
        "dumbbell_desc": (
            "Mean **Post-training DI** (± std across folds) for each intersectional subgroup. "
            "The gray line at DI=1.0 indicates perfect parity. "
            "Orange lines at 0.8 and 1.25 delimit the 80% Rule disparity zone."
        ),
        "post_di_label": "Post-training DI",
        "dumbbell_caption": (
            "🔴 DI < 0.8 (disadvantaged subgroup) | 🟠 DI > 1.25 (overrepresented) | 🔵 within acceptable range. "
            "Error bars = ±1 standard deviation across outer folds."
        ),
        "dumbbell_table_expander": "📋 Full data table by subgroup",
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
        "understand_metrics": "ℹ️ Understand the Metrics",
        "metrics_explanation": "* **Class Imbalance (CI)**: Measures demographic imbalance between the two extreme groups listed above. Varies from -1 (all samples in the unprivileged group) to 1 (all in the privileged group). Ideal: 0.\n* **Pre-training DI (Disparate Impact)**: Ratio between the chance of the unprivileged group belonging to the favorable target class and that of the privileged group. Values < 0.8 or > 1.2 indicate disparity (80% Rule).\n* **KL Divergence**: Measures the divergence (distance) between the probability distributions of outcomes for the privileged and unprivileged groups.\n* **KS Statistic (Kolmogorov-Smirnov)**: Measures the maximum distance between the cumulative distributions of the two groups. High values indicate inequality in the distribution.",
        
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
        "caption_inviable": "Subgroups with N < 100 are marked as 'Inviable'. Pre-training DI indicates Pre-training Disparate Impact against the global mean.",
        "export_csv": "📥 Export Data (Long Consolidated CSV)",
        
        "cddl_header": "Conditional Demographic Disparity in Labels (CDDL)",
        "no_proxy": "No proxy defined",
        "calc_cddl": "Calculating CDDL...",
        "caption_cddl": "Positive values indicate that the unprivileged class suffers more unfavorable results (and fewer favorable) conditioned to proxy strata.",
        "warn_metadata": "Incomplete metadata to define primary privileged/unprivileged group.",
        
        "audit_pairs": "Pairwise Gerrymandering Audit (Global)",
        "audit_pairs_desc": "This view scans **all possible pairs** between selected attributes to find the maximum disparity in the margin versus the intersection.",
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
        "tbl_gap_amplitude": "Gap Amplitude (%)"
    }
}
