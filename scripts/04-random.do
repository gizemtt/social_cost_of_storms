//------------------------------------------------------------------------------
// 04-randomization-placebo-county-linear.do
// County-level randomization placebo tests for compact Nature-lag linear model
// No future-lead placebo terms included here.
//------------------------------------------------------------------------------
//
// This script tests whether randomized TC exposure produces spurious SVI-specific
// TC mortality effects.
//
// Main randomized model:
//   tdths_ttpop = beta_low  * 1{svi_high=0} * X_nature_rnd
//              + beta_high * 1{svi_high=1} * X_nature_rnd
//              + fixed effects + error
//
// Randomization types:
//   1. within_monthyear : shuffle county exposure within each month
//   2. cross_panel      : assign each county another county's full exposure series
//   3. within_panel     : shuffle months within each county
//   4. total            : shuffle exposure across all county-month observations
//
// Important:
//   - Do not drop pre-1968 observations before constructing X_nature_rnd.
//   - The regression sample starts in 1968, but exposure history uses prior months.
//------------------------------------------------------------------------------


clear all
set more off
set seed 12345

//------------------------------------------------------------------------------
// User controls
//------------------------------------------------------------------------------

local DATA_IN  "data/DATA_hurricane_mortality_temp_month_county_19482018.dta"
local OUTDIR   "output/randomization_county_linear"

local Y        "tdths_ttpop"
local WGT      "totalpop"

local LMAX     172
local NSIM     1000

// Regression sample begins in 1968.
// Do not drop pre-1968 rows because X_nature uses up to 172 prior months.
local SAMPLE   "year >= 1968"

// SVI cutoff
local SVI_CUT  0.75

capture mkdir "`OUTDIR'"
capture mkdir "`OUTDIR'/logs"
capture mkdir "`OUTDIR'/results"

log using "`OUTDIR'/logs/04-randomization-placebo-county-linear.log", replace text


//------------------------------------------------------------------------------
// Package checks
//------------------------------------------------------------------------------

cap which reghdfe
if _rc {
    ssc install reghdfe, replace
}

cap which ftools
if _rc {
    ssc install ftools, replace
}


//------------------------------------------------------------------------------
// Step 1. Load and prepare base county-month data
//------------------------------------------------------------------------------

use "`DATA_IN'", clear

destring countyfips, replace force
destring stfips, replace force

sort countyfips modate

replace maxs = 0 if missing(maxs)

// Estimation sample only. Do not use keep if year >= 1968 before lag construction.
gen byte sample_ok = (`SAMPLE')
replace sample_ok = 0 if missing(`Y', maxs, tavg, `WGT', countyfips, modate, month, year)
replace sample_ok = 0 if `WGT' <= 0 | missing(`WGT')

//------------------------------------------------------------------------------
// Step 2. Construct SVI group
//------------------------------------------------------------------------------

bysort countyfips: egen mean_svi_county = mean(svi)

preserve
    keep countyfips mean_svi_county
    duplicates drop

    gen byte svi_high = .
    replace svi_high = 0 if mean_svi_county <= `SVI_CUT'
    replace svi_high = 1 if mean_svi_county >  `SVI_CUT' & mean_svi_county < .

    tempfile svi_groups
    save `svi_groups', replace
restore

merge m:1 countyfips using `svi_groups', nogen

replace sample_ok = 0 if missing(svi_high)

display "SVI group counts among estimation sample:"
tab svi_high if sample_ok


//------------------------------------------------------------------------------
// Step 3. Fixed-effect controls used in current model
//------------------------------------------------------------------------------

sum modate if sample_ok
replace modate = modate - round(r(mean))

forvalues p = 1/8 {
    capture drop modate_`p'
    gen double modate_`p' = modate^`p'
}

capture drop tavg_2 tavg_3
gen double tavg_2 = tavg^2
gen double tavg_3 = tavg^3

capture drop _im_
egen _im_ = group(countyfips month)


//------------------------------------------------------------------------------
// Step 4. Create balanced panel indexes for randomization
//------------------------------------------------------------------------------

egen county_ID = group(countyfips)
egen month_ID  = group(modate)

sum county_ID
local NCOUNTY = r(max)

sum month_ID
local NTIME = r(max)

display "Number of counties = `NCOUNTY'"
display "Number of months   = `NTIME'"

// The direct-index randomization below assumes a balanced county-month panel.
preserve
    keep county_ID month_ID
    duplicates drop

    bysort county_ID: gen n_time_by_county = _N
    quietly count if n_time_by_county != `NTIME'
    if r(N) > 0 {
        display as error "Panel is not balanced by county_ID x month_ID."
        display as error "Fix missing county-month rows or use a merge-based randomization."
        error 459
    }

    bysort month_ID: gen n_county_by_month = _N
    quietly count if n_county_by_month != `NCOUNTY'
    if r(N) > 0 {
        display as error "Panel is not balanced by month_ID x county_ID."
        display as error "Fix missing county-month rows or use a merge-based randomization."
        error 459
    }
restore

sort county_ID month_ID
gen long obs_index_panel = _n


//------------------------------------------------------------------------------
// Step 5. Precompute compact Nature lag weights
//------------------------------------------------------------------------------

scalar nature_sum_lag = 0

forvalues l = 0/`LMAX' {
    scalar nature_sum_lag = nature_sum_lag + ///
        (0.0237 + 0.000535*`l' - 0.0000039*(`l'^2))
}

display "Nature lag-shape sum, 0-`LMAX' = " nature_sum_lag

forvalues l = 0/`LMAX' {
    scalar g_lag`l' = ///
        (0.0237 + 0.000535*`l' - 0.0000039*(`l'^2)) / nature_sum_lag
}


//------------------------------------------------------------------------------
// Step 6. Helper program: build randomized compact exposure X_nature_rnd
//------------------------------------------------------------------------------

capture program drop build_compact_exposure
program define build_compact_exposure

    syntax, LMAX(integer)

    capture drop X_nature_rnd

    sort county_ID month_ID

    gen double X_nature_rnd = 0

    forvalues l = 0/`lmax' {
        tempvar lagval
        gen double `lagval' = .
        by county_ID: replace `lagval' = maxs_rnd[_n - `l']
        replace `lagval' = 0 if missing(`lagval')

        replace X_nature_rnd = X_nature_rnd + scalar(g_lag`l') * `lagval'
        drop `lagval'
    }

end


//------------------------------------------------------------------------------
// Step 7. Helper program: run randomized model
//------------------------------------------------------------------------------

capture program drop run_randomized_model
program define run_randomized_model, rclass

    quietly reghdfe tdths_ttpop ///
        ibn.svi_high#c.X_nature_rnd ///
        [aw=totalpop] if sample_ok, ///
        absorb( ///
            i.countyfips#i.month ///
            i.countyfips#c.modate_1 ///
            i._im_#c.modate_1 ///
            i.countyfips#c.tavg i.countyfips#c.tavg_2 ///
            i.modate ///
        ) ///
        vce(cluster countyfips) tol(1e-3)

    local b_low  = .
    local b_high = .
    local se_low = .
    local se_high = .

    capture local b_low = _b[0.svi_high#c.X_nature_rnd]
    capture local b_high = _b[1.svi_high#c.X_nature_rnd]
    capture local se_low = _se[0.svi_high#c.X_nature_rnd]
    capture local se_high = _se[1.svi_high#c.X_nature_rnd]

    return scalar b_low_actual  = `b_low'
    return scalar b_high_actual = `b_high'
    return scalar se_low_actual = `se_low'
    return scalar se_high_actual = `se_high'
    return scalar b_diff_high_low = `b_high' - `b_low'

    capture testparm ibn.svi_high#c.X_nature_rnd
    if !_rc {
        return scalar F_joint = r(F)
        return scalar p_joint = r(p)
    }
    else {
        return scalar F_joint = .
        return scalar p_joint = .
    }

    return scalar N    = e(N)
    return scalar r2_a = e(r2_a)

end


//------------------------------------------------------------------------------
// Step 8. Estimate actual model once and store actual coefficients
//------------------------------------------------------------------------------

display "------------------------------------------------------------"
display "Estimating actual model with observed X_nature"
display "------------------------------------------------------------"

capture drop X_nature
gen double X_nature = 0

sort county_ID month_ID

forvalues l = 0/`LMAX' {
    tempvar lagval_actual
    gen double `lagval_actual' = .
    by county_ID: replace `lagval_actual' = maxs[_n - `l']
    replace `lagval_actual' = 0 if missing(`lagval_actual')

    replace X_nature = X_nature + scalar(g_lag`l') * `lagval_actual'
    drop `lagval_actual'
}

reghdfe tdths_ttpop ///
    ibn.svi_high#c.X_nature ///
    [aw=totalpop] if sample_ok, ///
    absorb( ///
        i.countyfips#i.month ///
        i.countyfips#c.modate_1 ///
        i._im_#c.modate_1 ///
        i.countyfips#c.tavg i.countyfips#c.tavg_2 ///
        i.modate ///
    ) ///
    vce(cluster countyfips) tol(1e-3)

scalar actual_b_low  = _b[0.svi_high#c.X_nature]
scalar actual_b_high = _b[1.svi_high#c.X_nature]
scalar actual_se_low  = _se[0.svi_high#c.X_nature]
scalar actual_se_high = _se[1.svi_high#c.X_nature]
scalar actual_b_diff = actual_b_high - actual_b_low

testparm ibn.svi_high#c.X_nature
scalar actual_F_joint = r(F)
scalar actual_p_joint_asymptotic = r(p)

lincom 1.svi_high#c.X_nature - 0.svi_high#c.X_nature
scalar actual_diff_est = r(estimate)
scalar actual_diff_se = r(se)
scalar actual_diff_t = r(t)
scalar actual_diff_p_two = 2 * ttail(r(df), abs(r(t)))
scalar actual_diff_p_one_pos = ttail(r(df), r(t))

display "Actual low-SVI coefficient  = " actual_b_low
display "Actual high-SVI coefficient = " actual_b_high
display "Actual high-low difference  = " actual_b_diff
display "Actual joint F-statistic    = " actual_F_joint
display "Actual joint asymptotic p   = " actual_p_joint_asymptotic

tempfile actual_data
save `actual_data', replace


//------------------------------------------------------------------------------
// Step 9. Randomization loop
//------------------------------------------------------------------------------

tempfile base_data
save `base_data', replace

// Lookup table for randomized maxs assignment
preserve
    keep county_ID month_ID maxs
    rename county_ID draw_county_ID
    rename month_ID draw_month_ID
    rename maxs maxs_draw

    tempfile maxs_lookup
    save `maxs_lookup', replace
restore

// Lookup table for total randomization
preserve
    keep maxs
    rename maxs maxs_draw
    gen long draw_id = _n

    tempfile total_lookup
    save `total_lookup', replace
restore

quietly count
local NOBS = r(N)

local RANDTYPES "within_monthyear cross_panel within_panel total"

foreach randtype of local RANDTYPES {

    display "------------------------------------------------------------"
    display "Running randomization type: `randtype'"
    display "------------------------------------------------------------"

    postfile simout ///
        int sim ///
        str20 randtype ///
        double b_low_actual b_high_actual b_diff_high_low ///
        double se_low_actual se_high_actual ///
        double F_joint p_joint ///
        double N r2_a ///
        using "`OUTDIR'/results/randomization_`randtype'.dta", replace

    forvalues x = 1/`NSIM' {
        display "simulation run: `x'/`NSIM' for `randtype'"
        use `base_data', clear
        quietly {
            capture drop fake_county fake_month fake_index maxs_rnd maxs_draw
            capture drop draw_county_ID draw_month_ID draw_id
            capture drop X_nature_rnd

            // --------------------------------------------------------------
            // 1. Within-month-year randomization
            //    For each month, randomly assign exposure from another county
            //    in the same month.
            // --------------------------------------------------------------
            if "`randtype'" == "within_monthyear" {

                gen long draw_month_ID  = month_ID
                gen long draw_county_ID = floor(runiform() * `NCOUNTY') + 1

                merge m:1 draw_county_ID draw_month_ID using `maxs_lookup', ///
                    keep(master match) nogen

                gen double maxs_rnd = maxs_draw
            }

            // --------------------------------------------------------------
            // 2. Cross-panel randomization
            //    Assign each county another county's full exposure series.
            // --------------------------------------------------------------
            if "`randtype'" == "cross_panel" {

                gen long fake_county_temp = .
                bysort county_ID (month_ID): replace fake_county_temp = ///
                    floor(runiform() * `NCOUNTY') + 1 if _n == 1

                bysort county_ID: egen long draw_county_ID = max(fake_county_temp)
                gen long draw_month_ID = month_ID

                merge m:1 draw_county_ID draw_month_ID using `maxs_lookup', ///
                    keep(master match) nogen

                gen double maxs_rnd = maxs_draw

                drop fake_county_temp
            }

            // --------------------------------------------------------------
            // 3. Within-panel randomization
            //    For each county, randomly assign exposure from another month
            //    within the same county.
            // --------------------------------------------------------------
            if "`randtype'" == "within_panel" {

                gen long draw_county_ID = county_ID
                gen long draw_month_ID  = floor(runiform() * `NTIME') + 1

                merge m:1 draw_county_ID draw_month_ID using `maxs_lookup', ///
                    keep(master match) nogen

                gen double maxs_rnd = maxs_draw
            }

            // --------------------------------------------------------------
            // 4. Total randomization
            //    Randomly assign exposure from any county-month observation.
            // --------------------------------------------------------------
            if "`randtype'" == "total" {

                gen long draw_id = floor(runiform() * `NOBS') + 1

                merge m:1 draw_id using `total_lookup', ///
                    keep(master match) nogen

                gen double maxs_rnd = maxs_draw
            }

            replace maxs_rnd = 0 if missing(maxs_rnd)

            build_compact_exposure, lmax(`LMAX')

            run_randomized_model

            local rt "`randtype'"

            post simout ///
                (`x') ///
                (`"`rt'"') ///
                (r(b_low_actual)) ///
                (r(b_high_actual)) ///
                (r(b_diff_high_low)) ///
                (r(se_low_actual)) ///
                (r(se_high_actual)) ///
                (r(F_joint)) ///
                (r(p_joint)) ///
                (r(N)) ///
                (r(r2_a))
        }
    }

    postclose simout

    use "`OUTDIR'/results/randomization_`randtype'.dta", clear
    export delimited using "`OUTDIR'/results/randomization_`randtype'.csv", replace

}


//------------------------------------------------------------------------------
// Step 10. Combine randomization outputs
//------------------------------------------------------------------------------

clear
local first = 1

foreach randtype of local RANDTYPES {

    if `first' == 1 {
        use "`OUTDIR'/results/randomization_`randtype'.dta", clear
        local first = 0
    }
    else {
        append using "`OUTDIR'/results/randomization_`randtype'.dta"
    }
}

save "`OUTDIR'/results/randomization_all_types.dta", replace
export delimited using "`OUTDIR'/results/randomization_all_types.csv", replace


//------------------------------------------------------------------------------
// Step 11. Postprocessing: empirical randomization p-values
//------------------------------------------------------------------------------

use "`OUTDIR'/results/randomization_all_types.dta", clear

gen double actual_b_low  = scalar(actual_b_low)
gen double actual_b_high = scalar(actual_b_high)
gen double actual_b_diff = scalar(actual_b_diff)
gen double actual_F_joint = scalar(actual_F_joint)

// Two-sided empirical p-values
gen byte extreme_low_two  = abs(b_low_actual)  >= abs(actual_b_low)
gen byte extreme_high_two = abs(b_high_actual) >= abs(actual_b_high)
gen byte extreme_diff_two = abs(b_diff_high_low) >= abs(actual_b_diff)
gen byte extreme_joint_F  = F_joint >= actual_F_joint

// One-sided positive empirical p-values
gen byte extreme_low_pos  = b_low_actual  >= actual_b_low
gen byte extreme_high_pos = b_high_actual >= actual_b_high
gen byte extreme_diff_pos = b_diff_high_low >= actual_b_diff

// Save augmented simulation file
save "`OUTDIR'/results/randomization_all_types_with_extremes.dta", replace
export delimited using "`OUTDIR'/results/randomization_all_types_with_extremes.csv", replace


//------------------------------------------------------------------------------
// Step 12. Postprocessing table by randomization type
//------------------------------------------------------------------------------

tempfile pvals_by_type
postfile ptab ///
    str20 randtype ///
    int nsim ///
    double actual_b_low actual_b_high actual_b_diff actual_F_joint ///
    double mean_b_low mean_b_high mean_b_diff mean_F_joint ///
    double sd_b_low sd_b_high sd_b_diff sd_F_joint ///
    double p_low_two p_high_two p_diff_two p_joint_F ///
    double p_low_pos p_high_pos p_diff_pos ///
    using `pvals_by_type', replace

levelsof randtype, local(types)

foreach rt of local types {

    quietly count if randtype == "`rt'"
    local n = r(N)

    quietly summarize b_low_actual if randtype == "`rt'"
    local mean_low = r(mean)
    local sd_low = r(sd)

    quietly summarize b_high_actual if randtype == "`rt'"
    local mean_high = r(mean)
    local sd_high = r(sd)

    quietly summarize b_diff_high_low if randtype == "`rt'"
    local mean_diff = r(mean)
    local sd_diff = r(sd)

    quietly summarize F_joint if randtype == "`rt'"
    local mean_F = r(mean)
    local sd_F = r(sd)

    quietly summarize extreme_low_two if randtype == "`rt'"
    local p_low_two = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_high_two if randtype == "`rt'"
    local p_high_two = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_diff_two if randtype == "`rt'"
    local p_diff_two = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_joint_F if randtype == "`rt'"
    local p_joint_F = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_low_pos if randtype == "`rt'"
    local p_low_pos = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_high_pos if randtype == "`rt'"
    local p_high_pos = (r(sum) + 1) / (`n' + 1)

    quietly summarize extreme_diff_pos if randtype == "`rt'"
    local p_diff_pos = (r(sum) + 1) / (`n' + 1)

    post ptab ///
        ("`rt'") ///
        (`n') ///
        (scalar(actual_b_low)) ///
        (scalar(actual_b_high)) ///
        (scalar(actual_b_diff)) ///
        (scalar(actual_F_joint)) ///
        (`mean_low') ///
        (`mean_high') ///
        (`mean_diff') ///
        (`mean_F') ///
        (`sd_low') ///
        (`sd_high') ///
        (`sd_diff') ///
        (`sd_F') ///
        (`p_low_two') ///
        (`p_high_two') ///
        (`p_diff_two') ///
        (`p_joint_F') ///
        (`p_low_pos') ///
        (`p_high_pos') ///
        (`p_diff_pos')
}

postclose ptab

use `pvals_by_type', clear

save "`OUTDIR'/results/randomization_empirical_pvalues_by_type.dta", replace
export delimited using "`OUTDIR'/results/randomization_empirical_pvalues_by_type.csv", replace

list, abbreviate(24)


//------------------------------------------------------------------------------
// Step 13. Pooled empirical p-values across all randomization types
//------------------------------------------------------------------------------

use "`OUTDIR'/results/randomization_all_types_with_extremes.dta", clear

quietly count
local n_all = r(N)

quietly summarize extreme_low_two
local p_low_two_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_high_two
local p_high_two_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_diff_two
local p_diff_two_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_joint_F
local p_joint_F_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_low_pos
local p_low_pos_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_high_pos
local p_high_pos_all = (r(sum) + 1) / (`n_all' + 1)

quietly summarize extreme_diff_pos
local p_diff_pos_all = (r(sum) + 1) / (`n_all' + 1)

clear
set obs 1

gen str20 randtype = "pooled_all"
gen int nsim = `n_all'

gen double actual_b_low = scalar(actual_b_low)
gen double actual_b_high = scalar(actual_b_high)
gen double actual_b_diff = scalar(actual_b_diff)
gen double actual_F_joint = scalar(actual_F_joint)

gen double p_low_two   = `p_low_two_all'
gen double p_high_two  = `p_high_two_all'
gen double p_diff_two  = `p_diff_two_all'
gen double p_joint_F   = `p_joint_F_all'

gen double p_low_pos   = `p_low_pos_all'
gen double p_high_pos  = `p_high_pos_all'
gen double p_diff_pos  = `p_diff_pos_all'

save "`OUTDIR'/results/randomization_empirical_pvalues_pooled.dta", replace
export delimited using "`OUTDIR'/results/randomization_empirical_pvalues_pooled.csv", replace

list, abbreviate(24)

display "------------------------------------------------------------"
display "Done."
display "Key outputs:"
display "1. `OUTDIR'/results/randomization_all_types.csv"
display "2. `OUTDIR'/results/randomization_empirical_pvalues_by_type.csv"
display "3. `OUTDIR'/results/randomization_empirical_pvalues_pooled.csv"
display "------------------------------------------------------------"

log close