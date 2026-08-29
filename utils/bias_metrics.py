import pandas as pd
import numpy as np
import itertools
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

def calculate_cramer_v(df, col1, col2):
    """Calculates Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(df[col1], df[col2])
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def calculate_wilson_ci(successes, nobs, alpha=0.05):
    """Calculates Wilson score interval."""
    if nobs == 0:
        return 0, 0
    lower, upper = proportion_confint(count=successes, nobs=nobs, alpha=alpha, method='wilson')
    return lower, upper


def intersectional_audit_metrics(df, sensitive_attrs, target_col, favorable_val=1):
    """
    Calculates Gerrymandering Audit Metrics for intersectional subgroups.
    """
    global_rate = (df[target_col] == favorable_val).mean()
    
    # Calculate marginal rates
    marginal_rates = {}
    for attr in sensitive_attrs:
        marginal_rates[attr] = df.groupby(attr)[target_col].apply(lambda x: (x == favorable_val).mean()).to_dict()

    # Intersectional grouping
    groups = df.groupby(sensitive_attrs, observed=True)
    results = []

    for name, group in groups:
        n = len(group)
        if n == 0:
            continue
            
        # Handle single vs multiple sensitive attrs grouping properly
        if isinstance(name, tuple):
            subgroup_name = " & ".join([f"{attr}={val}" for attr, val in zip(sensitive_attrs, name)])
            vals = name
        else:
            subgroup_name = f"{sensitive_attrs[0]}={name}"
            vals = [name]
            
        favorable_count = (group[target_col] == favorable_val).sum()
        subgroup_rate = favorable_count / n
        
        real_gap = subgroup_rate - global_rate
        
        # Expected Gap: worst marginal gap among the intersecting identities
        marginal_gaps = []
        for attr, val in zip(sensitive_attrs, vals):
            m_rate = marginal_rates[attr].get(val, global_rate)
            marginal_gaps.append(m_rate - global_rate)
            
        expected_gap = min(marginal_gaps) if real_gap < 0 else max(marginal_gaps)
        
        hidden_bias = real_gap - expected_gap
        priority_score = hidden_bias * n
        
        # Pre-training DI
        di = subgroup_rate / global_rate if global_rate > 0 else 0
        
        # Audit Verdict
        if n < 100:
            verdict = "Inviável (N<100)"
        elif hidden_bias < -0.10:
            verdict = "Alto Viés Oculto"
        elif real_gap < -0.1:
            verdict = "Alto Viés Direto"
        else:
            verdict = "Ok"
            
        results.append({
            'Subgrupo': subgroup_name,
            'N': n,
            'Taxa Favorável': subgroup_rate,
            'Gap Real (Interseccional)': real_gap,
            'Gap Esperado (Marginal Máx)': expected_gap,
            'Viés Oculto (Excedente)': hidden_bias,
            'Score de Prioridade': priority_score,
            'DI Pré-treinamento': di,
            'Veredito da Auditoria': verdict
        })
        
    return pd.DataFrame(results)

def calculate_base_metrics(df, dataset_info):
    """
    Calculates the base metrics from Table 1 of the paper:
    - Class Dist (%)
    - CI Ratio
    - DI Pre-train
    """
    target_col = dataset_info['target']
    favorable_val = dataset_info['favorable_val']
    
    # Class Dist
    counts = df[target_col].value_counts()
    if len(counts) == 2:
        # Assuming binary target
        maj_count = counts.max()
        min_count = counts.min()
        total = maj_count + min_count
        maj_pct = (maj_count / total) * 100
        min_pct = (min_count / total) * 100
        class_dist = f"{maj_pct:.1f} / {min_pct:.1f}"
    else:
        class_dist = "N/A"
        
    # DI Pre-train and CI
    di_pre_train = "N/A"
    ci_metric = "N/A"
    primary_protected = dataset_info.get('primary_protected')
    privileged_group = dataset_info.get('privileged_group')
    unprivileged_group = dataset_info.get('unprivileged_group')
    di_target_val = dataset_info.get('di_target_val', favorable_val)
    
    if primary_protected and privileged_group and unprivileged_group and primary_protected in df.columns:
        priv_df = df[df[primary_protected] == privileged_group]
        unpriv_df = df[df[primary_protected] == unprivileged_group]
        
        n_p = len(priv_df)
        n_d = len(unpriv_df)
        
        if (n_p + n_d) > 0:
            ci_val = (n_p - n_d) / (n_p + n_d)
            ci_metric = f"{ci_val:.2f}"
            
        if len(priv_df) > 0 and len(unpriv_df) > 0:
            priv_rate = (priv_df[target_col] == di_target_val).mean()
            unpriv_rate = (unpriv_df[target_col] == di_target_val).mean()
            
            epsilon = 1e-9
            pp_1 = max(priv_rate, epsilon)
            pp_0 = max(1 - priv_rate, epsilon)
            pd_1 = max(unpriv_rate, epsilon)
            pd_0 = max(1 - unpriv_rate, epsilon)
            
            kl = pp_1 * np.log(pp_1 / pd_1) + pp_0 * np.log(pp_0 / pd_0)
            ks = max(abs(pp_1 - pd_1), abs(pp_0 - pd_0))
            
            if priv_rate > 0:
                di_pre_train = unpriv_rate / priv_rate
                di_pre_train = f"{di_pre_train:.2f}"
            else:
                di_pre_train = "N/A"
                
    return {
        'Distribuição de Classes (%)': class_dist
    }

def pairwise_gerrymandering_audit(df, attributes, target_col, favorable_val):
    """
    Varre todos os pares de atributos selecionados e compara a disparidade máxima marginal
    com a disparidade máxima interseccional.
    """
    results = []
    
    marginal_gaps = {}
    for attr in attributes:
        rates = df.groupby(attr)[target_col].apply(lambda x: (x == favorable_val).mean())
        if len(rates) > 1:
            marginal_gaps[attr] = rates.max() - rates.min()
        else:
            marginal_gaps[attr] = 0.0

    pairs = list(itertools.combinations(attributes, 2))
    
    for pair in pairs:
        attr_a, attr_b = pair
        gap_a = marginal_gaps[attr_a]
        gap_b = marginal_gaps[attr_b]
        
        expected_gap = max(gap_a, gap_b)
        
        inter_stats = df.groupby(list(pair), observed=True)[target_col].agg(
            rate=lambda x: (x == favorable_val).mean(),
            n='count'
        )
        
        viable_stats = inter_stats[inter_stats['n'] >= 100]
        
        if len(viable_stats) > 1:
            real_gap = viable_stats['rate'].max() - viable_stats['rate'].min()
        elif len(inter_stats) > 1:
            real_gap = inter_stats['rate'].max() - inter_stats['rate'].min()
        else:
            real_gap = 0.0
            
        hidden_bias = real_gap - expected_gap
        
        if hidden_bias > 0.10:
            verdict = "⚠️ GERRYMANDERING"
        else:
            verdict = "OK"
            
        results.append({
            'Par de Intersecção': f"{attr_a} × {attr_b}",
            'Gap Indiv. A': gap_a,
            'Gap Indiv. B': gap_b,
            'Gap Esperado (Marginal Máx)': expected_gap,
            'Gap Real (Interseccional)': real_gap,
            'Viés Oculto (Excedente)': hidden_bias,
            'Veredito da Auditoria': verdict
        })
        
    return pd.DataFrame(results)

def calculate_cddl(df, target_col, favorable_val, protected_attr, priv_group, unpriv_group, proxy_attrs):
    """
    Calculates Conditional Demographic Disparity in Labels (CDDL) for each proxy attribute.
    """
    results = []
    n_total = len(df)
    
    for proxy in proxy_attrs:
        cddl = 0.0
        
        # Agrupar pelos estratos do proxy
        for stratum, group in df.groupby(proxy, observed=True):
            n_i = len(group)
            if n_i == 0:
                continue
                
            priv_mask = (group[protected_attr] == priv_group)
            unpriv_mask = (group[protected_attr] == unpriv_group)
            
            fav_mask = (group[target_col] == favorable_val)
            unfav_mask = (group[target_col] != favorable_val)
            
            n_favorable = fav_mask.sum()
            n_unfavorable = unfav_mask.sum()
            
            n_d_favorable = (unpriv_mask & fav_mask).sum()
            n_d_unfavorable = (unpriv_mask & unfav_mask).sum()
            
            term1 = (n_d_unfavorable / n_unfavorable) if n_unfavorable > 0 else 0
            term2 = (n_d_favorable / n_favorable) if n_favorable > 0 else 0
            dd_i = term1 - term2
            
            cddl += (n_i * dd_i)
            
        cddl /= n_total
        results.append({
            'Atributo Protegido (Vítima)': protected_attr,
            'Grupo Desprivilegiado': unpriv_group,
            'Variável Proxy (Condição)': proxy,
            'CDDL': cddl
        })
        
    return pd.DataFrame(results)
def calculate_model_fairness_metrics(y_true, y_pred, groups_series, favorable_val=1):
    """
    Calculates intersectional post-training fairness metrics.

    This function evaluates the fairness of a trained classifier across all
    intersectional subgroups defined by `groups_series`. It is designed to be
    called once per outer fold of the nested cross-validation pipeline and
    then aggregated across folds.

    Metrics implemented (based on the reference paper):

    - **TPR** (True Positive Rate / Sensitivity / Recall):
        TPR = TP / (TP + FN)
        Measures the fraction of true positives correctly identified.

    - **FPR** (False Positive Rate):
        FPR = FP / (FP + TN)
        Measures the fraction of actual negatives incorrectly predicted as positive.

    - **Precision**:
        Precision = TP / (TP + FP)

    - **Post-training Disparate Impact (DI)**:
        DI_u = P(ŷ=favorable | group=u) / P(ŷ=favorable | group=reference)
        Where `reference` is the intersectional subgroup with the highest
        favorable prediction rate in this fold (determined dynamically).
        Values < 0.8 or > 1.25 are commonly considered disparate.

    - **AAOD** (Average Absolute Odds Difference) per subgroup:
        AAOD_u = 0.5 * (|FPR_u - FPR_ref| + |TPR_u - TPR_ref|)
        Measures the mean absolute deviation of both error rates from the
        reference group, capturing both under- and over-prediction simultaneously.

    - **Max Intersectional AAOD** (aggregate scalar):
        max_AAOD = max(AAOD_u for all viable subgroups)
        We use the maximum (not the mean) to adopt the most conservative stance
        and protect the worst-case subgroup, as recommended in high-stakes
        decision contexts.

    - **Sensitivity Gap** (aggregate scalar):
        sensitivity_gap = max(TPR_u) - min(TPR_u) for all viable subgroups (N >= 30)
        Measures the worst-case disparity in true positive rates.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Binary predictions from the trained model.
    groups_series : pd.Series
        Series of intersectional subgroup labels aligned with y_true and y_pred.
        Example values: "Female & Black", "Male & White".
    favorable_val : int or str, default=1
        The value considered the favorable outcome (positive class).

    Returns
    -------
    subgroup_metrics : pd.DataFrame
        One row per subgroup with columns:
        ['subgroup', 'n', 'tpr', 'fpr', 'precision', 'favorable_rate', 'post_di', 'aaod']
    aggregate_metrics : dict
        Scalar summary metrics:
        {'max_aaod', 'sensitivity_gap', 'reference_group'}
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    groups = pd.Series(groups_series).reset_index(drop=True)

    unique_groups = groups.unique()
    results = []

    for group in unique_groups:
        mask = groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        n = mask.sum()

        if n == 0:
            continue

        fav_mask_true = (yt == favorable_val)
        fav_mask_pred = (yp == favorable_val)

        tp = int((fav_mask_pred & fav_mask_true).sum())
        fn = int((~fav_mask_pred & fav_mask_true).sum())
        fp = int((fav_mask_pred & ~fav_mask_true).sum())
        tn = int((~fav_mask_pred & ~fav_mask_true).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        favorable_rate = fav_mask_pred.mean()

        results.append({
            'subgroup': group,
            'n': n,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'favorable_rate': favorable_rate,
            # DI and AAOD computed after reference group is identified below
            'post_di': np.nan,
            'aaod': np.nan,
        })

    if not results:
        empty_df = pd.DataFrame(columns=['subgroup', 'n', 'tpr', 'fpr', 'precision',
                                         'favorable_rate', 'post_di', 'aaod'])
        return empty_df, {'max_aaod': np.nan, 'sensitivity_gap': np.nan, 'reference_group': None}

    subgroup_df = pd.DataFrame(results)

    # Reference group: highest favorable prediction rate (determined dynamically)
    ref_idx = subgroup_df['favorable_rate'].idxmax()
    ref_group = subgroup_df.loc[ref_idx, 'subgroup']
    ref_favorable_rate = subgroup_df.loc[ref_idx, 'favorable_rate']
    ref_tpr = subgroup_df.loc[ref_idx, 'tpr']
    ref_fpr = subgroup_df.loc[ref_idx, 'fpr']

    # Compute DI and AAOD for each subgroup
    subgroup_df['post_di'] = subgroup_df['favorable_rate'].apply(
        lambda r: r / ref_favorable_rate if ref_favorable_rate > 0 else np.nan
    )
    subgroup_df['aaod'] = subgroup_df.apply(
        lambda row: 0.5 * (abs(row['fpr'] - ref_fpr) + abs(row['tpr'] - ref_tpr)),
        axis=1
    )

    # Aggregate: viable subgroups only (N >= 30) for sensitivity gap
    viable = subgroup_df[subgroup_df['n'] >= 30]
    sensitivity_gap = (viable['tpr'].max() - viable['tpr'].min()) if len(viable) >= 2 else np.nan
    max_aaod = subgroup_df['aaod'].max()

    aggregate_metrics = {
        'max_aaod': max_aaod,
        'sensitivity_gap': sensitivity_gap,
        'reference_group': ref_group,
    }

    return subgroup_df, aggregate_metrics

def calculate_dynamic_metrics(df, attr, target_col, favorable_val):
    """
    Calcula as métricas de viés (CI, DI, KL, KS) dinamicamente para um dado atributo.
    Identifica o grupo com maior taxa de sucesso como 'privilegiado' 
    e o de menor taxa como 'desprivilegiado'.
    Para o CI, usa a diferença de tamanho entre esses dois mesmos grupos extremos.
    """
    groups = df[attr].dropna().unique()
    if len(groups) < 2:
        return {'CI': 'N/A', 'DI': 'N/A', 'KL': 'N/A', 'KS': 'N/A', 'priv': None, 'unpriv': None}
        
    # Calcular taxas de sucesso
    rates = {}
    sizes = {}
    for g in groups:
        g_df = df[df[attr] == g]
        sizes[g] = len(g_df)
        if len(g_df) > 0:
            rates[g] = (g_df[target_col] == favorable_val).mean()
        else:
            rates[g] = 0.0
            
    # Encontrar priviliegiado e desprivilegiado
    priv_group = max(rates, key=rates.get)
    unpriv_group = min(rates, key=rates.get)
    
    # Se todas as taxas forem iguais, fallback para tamanho
    if rates[priv_group] == rates[unpriv_group]:
        priv_group = max(sizes, key=sizes.get)
        unpriv_group = min(sizes, key=sizes.get)
        
    n_p = sizes[priv_group]
    n_d = sizes[unpriv_group]
    
    ci_metric = "N/A"
    if (n_p + n_d) > 0:
        ci_val = (n_p - n_d) / (n_p + n_d)
        ci_metric = f"{ci_val:.2f}"
        
    priv_rate = rates[priv_group]
    unpriv_rate = rates[unpriv_group]
    
    epsilon = 1e-9
    pp_1 = max(priv_rate, epsilon)
    pp_0 = max(1 - priv_rate, epsilon)
    pd_1 = max(unpriv_rate, epsilon)
    pd_0 = max(1 - unpriv_rate, epsilon)
    
    kl = pp_1 * np.log(pp_1 / pd_1) + pp_0 * np.log(pp_0 / pd_0)
    ks = max(abs(pp_1 - pd_1), abs(pp_0 - pd_0))
    
    di_pre_train = "N/A"
    if priv_rate > 0:
        di_val = unpriv_rate / priv_rate
        di_pre_train = f"{di_val:.2f}"
        
    return {
        'CI': ci_metric,
        'DI': di_pre_train,
        'KL': f"{kl:.3f}",
        'KS': f"{ks:.3f}",
        'priv': priv_group,
        'unpriv': unpriv_group
    }
