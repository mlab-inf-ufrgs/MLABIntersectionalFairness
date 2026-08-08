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
        "models_page_title": "Modelos (Em Breve)",
        "models_title": "Modelos Pós-Treinamento",
        "models_info": "**Aba em desenvolvimento.**\n\nNesta etapa futura, adicionaremos os resultados de modelos (RF, GBM, MLP) treinados sobre os datasets selecionados.\nAs métricas de auditoria incluirão:\n- Sensitivity Gap\n- Average Absolute Odds Difference (AAOD)\n- Disparate Impact (Pós-treinamento)\n\nPor enquanto, utilize a aba **Dados** para análise exploratória e diagnóstico de viés inerente (pré-treinamento).",
        
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
        "models_page_title": "Models (Coming Soon)",
        "models_title": "Post-Training Models",
        "models_info": "**Tab in development.**\n\nIn this future stage, we will add the results of models (RF, GBM, MLP) trained on the selected datasets.\nAudit metrics will include:\n- Sensitivity Gap\n- Average Absolute Odds Difference (AAOD)\n- Disparate Impact (Post-training)\n\nFor now, use the **Data** tab for exploratory analysis and inherent bias diagnosis (pre-training).",
        
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
