import pandas as pd
import numpy as np
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

def calculate_model_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    To be used in the 'Modelos' tab later.
    Calculates Sensitivity Gap and AAOD.
    """
    pass
