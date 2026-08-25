# Social Vulnerability and Long-Run Tropical Cyclone Mortality

Replication code for **“Social Vulnerability and Long-Run Tropical Cyclone Mortality: Evidence from Texas.”** Stata prepares the data and estimates the models; Python performs validation and creates the figures.

## Repository structure

```text
social_cost_of_storms/
├── data/
│   ├── DATA_hurricane_mortality_temp_month_county_19482018.dta
│   ├── panel_by_storm__NA_USA_density_8_yr_1930_2018.csv
│   └── county_boundaries/
│       └── tl_2018_us_county/
├── output/
│   ├── beta/
│   ├── figures/
│   │   └── resid_plots/
│   └── randomization_county_linear/
├── scripts/
│   ├── master.do
│   ├── 01-prep-lags.do
│   ├── 02-run-regressions.do
│   ├── 03-postprocess-coefficients.do
│   ├── 04-random.do
│   ├── validation_postprocessing.py
│   └── plot_*.py
└── README.md
```

Download the archived inputs from Zenodo and place them under `data/`. Keep all generated files under `output/`.

## Requirements

- Stata 18 SE with `reghdfe`, `ftools`, and `parmest`
- Python 3.10+
- Python packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, `scikit-learn`, and `geopandas`

## Run the main analysis

`master.do` uses relative paths and must be launched from the repository root. In Windows Command Prompt, run:

```bat
cd /d "your_path\social_cost_of_storms"
"C:\Program Files\Stata18\StataSE-64.exe" /b do "scripts\master.do"
```

In PowerShell, run:

```powershell
cd "your_path\social_cost_of_storms"
& "C:\Program Files\Stata18\StataSE-64.exe" /b do "scripts\master.do"
```

Replace `your_path` with the directory containing the cloned repository.

`master.do` runs:

1. `01-prep-lags.do`: constructs the 0–172-month observed-exposure index and the 1–12-month future-placebo index.
2. `02-run-regressions.do`: estimates the population-weighted SVI-group model and exports residuals and fitted values.
3. `03-postprocess-coefficients.do`: exports the coefficients and variance-covariance matrix.

Principal outputs:

```text
output/full_data_for_regression.dta
output/beta/m_quadpoly_tdths_ttpop.ster
output/beta/residuals_fitted.csv
output/beta/coefficients_county_nature_future.dta
output/beta/coefficients_region_poly.csv
output/beta/vcv_region_poly.csv
```

## Randomization tests

`04-random.do` is not called by `master.do`. Run it separately from the repository root:

```bat
"C:\Program Files\Stata18\StataSE-64.exe" /b do "scripts\04-random.do"
```

It runs 1,000 replications for each of four exposure-randomization designs. Results are saved under:

```text
output/randomization_county_linear/
```

## Validation

Run after the main Stata analysis:

```bat
python scripts\validation_postprocessing.py
```

The script evaluates residual patterns, heteroskedasticity, seasonality, aggregate and within-county autocorrelation, aggregate-residual stationarity, and event-time residual behavior around wind exposure of at least 25 m/s.

Validation plots and supporting files are saved under:

```text
output/figures/resid_plots/
```

The manuscript diagnostic panel combines:

```text
standardized_residuals_vs_fitted.png
aggregate_monthly_mean_residual.png
acf_aggregate_monthly_mean_residuals.png
residual_event_time_check_maxs25.png
```

## Figures

Python figure scripts are run separately and individually after the Stata analysis:

```bat
python scripts\plot_beta_estimates.py
python scripts\plot_randomization_mean_beta_estimates.py
python scripts\plot_resid_panels.py
python scripts\plot_storm_burden.py
python scripts\plot_storm_wind_vs_tc_mortality_172mo.py
python scripts\plot_SVI_to_mortality.py
python scripts\plot_tx_mortality_map.py
```

Figures and plotting-data files are saved under:

```text
output/figures/
```

## Data availability

Code is available in this GitHub repository. Input and replication datasets are archived separately on [Zenodo](https://doi.org/10.5281/zenodo.22097263).

## Citation
Cite the accompanying manuscript and the Zenodo dataset record. Full publication information will be added after acceptance.
