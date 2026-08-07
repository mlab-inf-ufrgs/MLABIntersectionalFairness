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
            verdict = "Inviable (N<100)"
        elif hidden_bias < -0.05:
            verdict = "High Hidden Bias"
        elif real_gap < -0.1:
            verdict = "High Direct Bias"
        else:
            verdict = "Ok"
            
        results.append({
            'Subgroup': subgroup_name,
            'N': n,
            'Favorable Rate': subgroup_rate,
            'Real Gap (Intersectional)': real_gap,
            'Expected Gap (Max Marginal)': expected_gap,
            'Hidden Bias (Surplus)': hidden_bias,
            'Priority Score': priority_score,
            'Pre-training DI': di,
            'Audit Veredict': verdict
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
        'Class Distribution (%)': class_dist,
        'Class Imbalance (CI)': ci_metric,
        'DI Pre-train': di_pre_train,
        'KL Divergence': f"{kl:.3f}" if 'kl' in locals() else "N/A",
        'KS Statistic': f"{ks:.3f}" if 'ks' in locals() else "N/A"
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
        
        if hidden_bias > 0.05:
            verdict = "⚠️ GERRYMANDERING"
        else:
            verdict = "OK"
            
        results.append({
            'Intersection Pair': f"{attr_a} × {attr_b}",
            'Indiv. Gap A': gap_a,
            'Indiv. Gap B': gap_b,
            'Expected Gap (Max Marginal)': expected_gap,
            'Real Gap (Intersectional)': real_gap,
            'Hidden Bias (Surplus)': hidden_bias,
            'Audit Veredict': verdict
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
            'Proxy (Estrato)': proxy,
            'CDDL': cddl
        })
        
    return pd.DataFrame(results)
def calculate_model_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    To be used in the 'Modelos' tab later.
    Calculates Sensitivity Gap and AAOD.
    """
    pass
