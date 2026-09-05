import numpy as np
import pandas as pd
from scipy import stats
from copulas.multivariate import GaussianMultivariate
def generate_syn_sample(real_df, n_sample=500):
    """Copula生成硅基虚拟样本"""
    cols = ["urban_rate", "pri_ratio", "finance_self_support", "gdp_per_capita"]
    real_data = real_df[cols].dropna().copy()
    copula = GaussianMultivariate()
    copula.fit(real_data)
    syn_data = copula.sample(n_sample)
    syn_df = pd.DataFrame(syn_data, columns=cols)
    return real_data, syn_df
def validate_syn_sample(real_df, syn_df):
    """硅基样本有效性校验：KS检验、均值、方差相对偏差"""
    res = {}
    cols = real_df.columns.tolist()
    for col in cols:
        ks_stat, ks_p = stats.ks_2samp(real_df[col], syn_df[col])
        mean_real = real_df[col].mean()
        mean_syn = syn_df[col].mean()
        var_real = real_df[col].var()
        var_syn = syn_df[col].var()

        mean_rel_err = abs(mean_real - mean_syn)/mean_real if mean_real !=0 else np.nan
        var_rel_err = abs(var_real - var_syn)/var_real if var_real !=0 else np.nan

        res[col] = {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "真实样本均值": mean_real,
            "硅基样本均值": mean_syn,
            "均值相对偏差": mean_rel_err,
            "真实样本方差": var_real,
            "硅基样本方差": var_syn,
            "方差相对偏差": var_rel_err
        }
    return res