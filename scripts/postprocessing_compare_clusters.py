import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = r"C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms"
# =============================================================================
# # Directories representing different clustering levels
# dirs = [os.path.join(base_dir, 'mycode-mydataset-stateCluster'), os.path.join(base_dir,'Only wind speed clustered - 90percent of max height'), os.path.join(base_dir,'onlywind_cluster_35percent_of_max_height')]  # Add paths as needed
# cluster_counts = [35, 71, 228]  # Must align with `dirs`
# =============================================================================

# =============================================================================
# dirs = [os.path.join(base_dir, 'mycode-mydataset-stateCluster')]  # os.path.join(base_dir, 'mycode-theirfeatures-state')
# cluster_counts = [35]  # Must align with `dirs`
# =============================================================================

dirs = [os.path.join(base_dir,'Only wind speed clustered - 90percent of max height'), os.path.join(base_dir,'90withighertermsstatelevel')]  # Add paths as needed
cluster_counts = [71, 70]  # Must align with `dirs`

# Initialize lists
r2_vals = []
sig_lag_perc = [] 
beta_dfs = []
comp_times = []

for d, cluster in zip(dirs, cluster_counts):
    
    with open(os.path.join(d, 'master.log'), 'r', encoding='utf-8') as f:
        log = f.read()

    # Extract coefficients and p-values
    coeffs = re.findall(r'_LL(\d+)_maxs\s+\|\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+[-\d\.]+\s+([-\d\.eE]+)', log)
    df_beta = pd.DataFrame(coeffs, columns=['Lag', 'Coef', 'SE', 'P'])
    df_beta['Lag'] = df_beta['Lag'].astype(int)
    df_beta['Coef'] = df_beta['Coef'].astype(float)
    df_beta['P'] = df_beta['P'].astype(float)
    df_beta['SE'] = df_beta['SE'].astype(float)
    df_beta['Cluster'] = cluster
    beta_dfs.append(df_beta)

    # R² extraction
    match_r2 = re.search(r'R-squared\s+=\s+([\d\.]+)', log)
    r2_vals.append(float(match_r2.group(1)) if match_r2 else None)

    # % Significant Lags (P < 0.05)
    pct_sig = (df_beta['P'] < 0.05).sum() / len(df_beta)
    sig_lag_perc.append(pct_sig * 100)
    
    # Look for the last non-empty line containing computational time
    timer1 = None
    for line in reversed(log.splitlines()):
        if line.strip().startswith('1:') and '/' in line:
            try:
                timer1 = float(line.strip().split(':')[1].split('/')[0])
                break
            except ValueError:
                continue
    comp_times.append(timer1)


# Merge β coefficients
beta_all = pd.concat(beta_dfs)


# --- New Section: Generate Summary Table ---
start_year = 1953
end_year = 2018
lags =  180
summary = []

# already have beta_all merged across clusters
for cluster in cluster_counts:
    # Filter for the current cluster
    sub_beta = beta_all[beta_all['Cluster'] == cluster]
    
    # --- Separate plot for each cluster: β vs Lag with Confidence Intervals ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sub_beta['Lag'], sub_beta['Coef'],
        yerr=1.96 * sub_beta['SE'],  # 95% confidence interval
        fmt='o-',               # points + line
        capsize=4,              # add little caps to error bars
        label=f'{cluster} clusters'
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Lag (Months)')
    plt.ylabel('β Coefficient')
    plt.title(f'β vs Lag with 95% CI - {cluster} Clusters')
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.06, 0.08)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------- #

    # Load exposure matrix
    str_df = pd.read_stata(os.path.join(dirs[cluster_counts.index(cluster)],'beta' , 'm_linear_tdths_ttpop.dta'))
    str_df = str_df[(str_df['year']>=start_year) & (str_df['year']<=end_year)]
    
    regression_df = pd.read_stata(os.path.join(dirs[cluster_counts.index(cluster)], 'full_data_for_regression.dta'))
    regression_df = regression_df.drop(columns=[col for col in regression_df.columns if col.startswith('_FF') ])
    regression_df = regression_df[(regression_df['year']>=start_year) & (regression_df['year']<=end_year)]
    
    # Replace missing exposures with 0
    for col in regression_df.columns:
        if col.startswith('_LL') and col.endswith('_maxs'):
            regression_df[col] = regression_df[col].fillna(0)
        
    # First, merge the two datasets
    merged_df = pd.merge(
        str_df[['year', 'month', 'cluster_label'] + [col for col in str_df.columns if col.startswith('beta_L')]],
        regression_df,
        how='inner',
        on=['year', 'month', 'cluster_label']
    )
    
          
    # -------------------------------------------- # 
    # Prediction Interval Computation
    
    # Step 1: Create matrix of exposures for all rows (rows = observations, cols = lags)
    lag_cols = [f'_LL{lag}_maxs' for lag in range(0, lags + 1)]
    exposure_matrix = merged_df[lag_cols].values  # shape: (n_obs, n_lags)
    
    # Step 2: Get cluster-level betas aligned with lag order
    beta_cols = [f'beta_L{lag}_maxs' for lag in range(0, lags + 1)]
    beta_vector = merged_df[beta_cols].values.T.mean(axis=1)  # average over all observations (shape: lags,)
    # beta_l . Change if you turn it to beta_li !!!!!!
    
    # Step 3: Calculate variance for each row's predicted deaths
    # mortality = pop * exposure * beta → variance is (Xβ)' V (Xβ), scaled by (pop/100000)^2
    pop_scale = (merged_df['totalpop'] / 100000).values.reshape(-1, 1)  # shape: (n_obs, 1)
    
    X_beta = exposure_matrix * beta_vector * pop_scale  # shape: (n_obs, n_lags)
    
    # point estimate: Sum across lags → row-wise sum for each observation
    merged_df['total_predicted_deaths'] = X_beta.sum(axis=1) # shape (n_obs,)
    deaths_point = merged_df['total_predicted_deaths'].sum()
    print("total_predicted_deaths= ", deaths_point)
    
    # point estimate (truncated)
    X_beta = np.clip(X_beta, a_min=0, a_max=None)    # Clip negative values to zero
    merged_df['neg_truncated_total_predicted_deaths'] = X_beta.sum(axis=1) # shape (n_obs,)
    deaths_point_trunc = merged_df['neg_truncated_total_predicted_deaths'].sum()
    print("negatives_truncated_total_predicted_deaths =", deaths_point_trunc)
   
# =============================================================================
#     # Load corresponding mortality prediction file (to compare with stata output, temporary)
#     mort_file = os.path.join(dirs[cluster_counts.index(cluster)], 'mortality_predict_state.csv')
#     mort_df = pd.read_csv(mort_file)
#     print(merged_df['total_predicted_deaths'].sum(), f"for {cluster} clusters while stata output is {mort_df['mort'].sum()}")    
# =============================================================================
     
    # Step 4: Compute 95% CI assuming normality
    
    # Load the variance-covariance matrix
    vcv_path = os.path.join(dirs[cluster_counts.index(cluster)], 'vcv.csv')
    vcv = pd.read_csv(vcv_path, header=0)
    vcv.columns = [col.strip() for col in vcv.columns]  # remove accidental spaces
    vcv.index = vcv.columns.copy()
    
    # Variance-covariance matrix must match lag order
    vcv_matrix = vcv.loc[beta_cols, beta_cols].values  # (lags x lags)
    
    # Calculate row-wise variances: diag(X @ V @ X.T)
    X = exposure_matrix * pop_scale  # shape: (n_obs, n_lags)
    variances = np.einsum('ij,jk,ik->i', X, vcv_matrix, X)  # shape: (n_obs,)
    
    # Step 5: Calculate standard errors and confidence bounds
    stderr = np.sqrt(variances)
    z = 1.96  # for 95% confidence
    merged_df["deaths_lower"] = merged_df["total_predicted_deaths"] - z * stderr
    merged_df["deaths_upper"] = merged_df["total_predicted_deaths"] + z * stderr
    
    # ------ #
    '''
    When we do $\beta_{li}$, we will have separate variance covariance matrix for each region i. 
    Then, computation will change slightly.
    '''
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
        'Cluster Count': cluster,
        'R_squared': r2_vals[cluster_counts.index(cluster)],
        '% Significant Lags': sig_lag_perc[cluster_counts.index(cluster)],
        'Deaths Point Estimate (Truncated)': deaths_point_trunc,
        'Deaths Point Estimate': deaths_point,
        'Deaths 95% CI Lower': deaths_lower,
        'Deaths 95% CI Upper': deaths_upper,
        'Computation Time (sec)': comp_times[cluster_counts.index(cluster)]
    })
    
    # -------------------------------------------- #
    # Save Summary CSV
    summary_df = pd.DataFrame(summary)
        
    # Drop lag/beta columns to reduce memory usage
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.startswith('_LL') or col.startswith('beta_L')])

summary_df.to_csv(os.path.join(base_dir, 'cluster_comparison_summary.csv'), index=False)
# %%
# Plot prediction intervals

x = summary_df['Cluster Count']
y = summary_df['Deaths Point Estimate']
yerr_lower = y - summary_df['Deaths 95% CI Lower']
yerr_upper = summary_df['Deaths 95% CI Upper'] - y
yerr = [yerr_lower, yerr_upper]

plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=yerr, fmt='o', linestyle='--', capsize=5, label='Point Estimate ± 95% CI')
for xi, yi in zip(x, y):
    plt.text(xi, yi, str(xi), ha='right', va='bottom', fontsize=8)

plt.xlabel('Number of Clusters')
plt.ylabel('Total Predicted Deaths')
plt.title('Predicted Deaths and 95% Confidence Intervals by Cluster Granularity')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()




    
