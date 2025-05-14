'''
postprocessing for beta_li version
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

base_dir = r"C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms"
output_dir = os.path.join(base_dir,'state_beta_l_i_incomplete') #output
beta_dir = os.path.join(output_dir,'beta')

num_month_aggragation = 1
num_lags = int(180 / num_month_aggragation)

start_year = 1968
end_year = 2018
# --------------------------------------------------------------- #

regression_df = pd.read_stata(os.path.join(output_dir, 'full_data_for_regression.dta'))
regression_df = regression_df.drop(columns=[col for col in regression_df.columns if col.startswith('_FF') ])
regression_df = regression_df[(regression_df['year']>=start_year) & (regression_df['year']<=end_year)]

# Replace missing exposures with 0
for col in regression_df.columns:
    if col.startswith('_LL') and col.endswith('_maxs'):
        regression_df[col] = regression_df[col].fillna(0)
    
# --------------------------------------------------------------- #

# Load full VCV matrix
df_vcv= pd.read_csv(os.path.join(output_dir,"vcv_beta_li_full.csv") , index_col=0)

# Load beta table and extract countyfips, lag
df_beta = pd.read_stata(os.path.join(output_dir, "beta_li_output.dta"))

def extract_parts(parm):
    parm_clean = parm.replace('b.', '.')  # normalize Stata base level labels
    match = re.match(r"(\d+)\.countyfips#c\._LL(\d+)_maxs", parm_clean)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

df_beta[['countyfips', 'lag']] = df_beta['parm'].apply(lambda x: pd.Series(extract_parts(x)))
df_beta = df_beta.dropna(subset=['countyfips', 'lag'])
df_beta['countyfips'] = df_beta['countyfips'].astype(int)
df_beta['lag'] = df_beta['lag'].astype(int)

# Extract unique counties
unique_fips = df_beta['countyfips'].unique()
final = []
summary = []
# Initialize national totals
national_xtotal = np.zeros(num_lags + 1)
national_variance = 0.0
national_deaths_point = 0.0
# === Export VCV block for each county ===
for fips in unique_fips:
    sub_beta = df_beta[df_beta['countyfips'] == fips] # Subset beta rows
    sub_reg = regression_df[regression_df['countyfips'] == fips].copy()  # Subset regression rows
    
    
    # -------------------------------------------- # 
    # Prediction Interval Computation
    
    # Step 1: Create matrix of exposures for all rows (rows = observations, cols = lags)
    lag_cols = [f'_LL{lag}_maxs' for lag in range(0, num_lags + 1)]
    exposure_matrix = sub_reg[lag_cols].values  # shape: (n_obs, n_lags)
    
    # Step 2: Get cluster-level betas aligned with lag order
    sub_beta = sub_beta.sort_values('lag')
    beta_vector = sub_beta['estimate'].values  # Shape: (n_lags,)

    # Step 3: Calculate variance for each row's predicted deaths
    # mortality = pop * exposure * beta → variance is (Xβ)' V (Xβ), scaled by (pop/100000)^2
    pop_scale = (sub_reg['totalpop'] / 100000).values.reshape(-1, 1)  # shape: (n_obs, 1)
    
    X_beta = exposure_matrix * beta_vector * pop_scale  # shape: (n_obs, n_lags)
    
    # point estimate: Sum across lags → row-wise sum for each observation
    sub_reg['total_predicted_deaths'] = X_beta.sum(axis=1) # shape (n_obs,)
    deaths_point = sub_reg['total_predicted_deaths'].sum()
    print(f"total_predicted_deaths for cluster {fips}= {deaths_point}")
    
    # point estimate (truncated)
    X_beta = np.clip(X_beta, a_min=0, a_max=None)    # Clip negative values to zero
    sub_reg['neg_truncated_total_predicted_deaths'] = X_beta.sum(axis=1) # shape (n_obs,)
    deaths_point_trunc = sub_reg['neg_truncated_total_predicted_deaths'].sum()
    print("negatives_truncated_total_predicted_deaths =", deaths_point_trunc)
    
    # Step 4: Compute 95% CI assuming normality
    # Load the variance-covariance matrix
    matching_labels = df_beta.loc[df_beta['countyfips'] == fips, 'parm'].tolist() # Filter relevant β names for this fips
    # Variance-covariance matrix must match lag order
    vcv_matrix = df_vcv.loc[matching_labels, matching_labels].values # Subset VCV rows and columns # (lags x lags)
    
    # Calculate row-wise variances: diag(X @ V @ X.T)
    X = exposure_matrix * pop_scale  # shape: (n_obs, n_lags)
    variances = np.einsum('ij,jk,ik->i', X, vcv_matrix, X)  # shape: (n_obs,)
    
    xtotal = X.sum(axis=0) # for national variance calculation
    national_variance += xtotal @ vcv_matrix @ xtotal.T
 
    # Step 5: Calculate standard errors and confidence bounds
    stderr = np.sqrt(variances)
    z = 1.96  # for 95% confidence
    sub_reg["deaths_lower"] = sub_reg["total_predicted_deaths"] - z * stderr
    sub_reg["deaths_upper"] = sub_reg["total_predicted_deaths"] + z * stderr
  
    # ------ #
    # Compute total variance and confidence interval for total predicted deaths
    
    # Replace any residual NaNs with 0 before aggregation
    X = np.nan_to_num(X, nan=0.0)
    
    # Sum exposure over all observations
    xtotal = X.sum(axis=0)  # shape: (n_lags,)
    
    total_variance = xtotal @ vcv_matrix @ xtotal.T
    total_stderr = np.sqrt(total_variance)
    
    z = 1.96  # for 95% CI
    deaths_lower = deaths_point - z * total_stderr
    deaths_upper = deaths_point + z * total_stderr

    print("deaths_lower =", deaths_lower)
    print("deaths_upper =", deaths_upper)
    
    # -------------------------------------------- #
    # Add to Summary Table
    summary.append({
        'countyfips': fips,
        #'% Significant Lags': sig_lag_perc[cluster_counts.index(cluster)],
        'Deaths Point Estimate (Truncated)': deaths_point_trunc,
        'Deaths Point Estimate': deaths_point,
        'Deaths 95% CI Lower': deaths_lower,
        'Deaths 95% CI Upper': deaths_upper,
    })
    
    # -------------------------------------------- #
    
    
    # Drop lag/beta columns to reduce memory usage
    sub_reg = sub_reg.drop(columns=[col for col in sub_reg.columns if col.startswith('_LL')])
    final.append(sub_reg)


# Save CSV
summary_df = pd.DataFrame(summary)

# Compute national standard error and CI
national_deaths_point = summary_df['Deaths Point Estimate'].sum()
national_stderr = np.sqrt(national_variance)
z = 1.96
lower = national_deaths_point - z * national_stderr
upper = national_deaths_point + z * national_stderr

print("\nNATIONAL-LEVEL TOTAL")
print("Total Point Estimate:", national_deaths_point)
print("95% CI:", (lower, upper))

summary.append({
    'countyfips': 'total',
    #'% Significant Lags': sig_lag_perc[cluster_counts.index(cluster)],
    'Deaths Point Estimate (Truncated)': summary_df['Deaths Point Estimate (Truncated)'].sum(),
    'Deaths Point Estimate': national_deaths_point,
    'Deaths 95% CI Lower': lower,
    'Deaths 95% CI Upper': upper,
})


'''
Add R_squared and computation time from log here. Append to summary!
'''

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)  

df_final = pd.concat(final, ignore_index=True)
df_final.to_csv(os.path.join(output_dir, 'detailed_estimations.csv'), index=False) 

