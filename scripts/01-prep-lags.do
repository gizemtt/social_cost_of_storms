//------------------------------------------------------------------------------
// prepping temp, lag, and placebo lead variables
//------------------------------------------------------------------------------

// monthly data during 1948-2018
use data/DATA_hurricane_mortality_temp_month_county_19482018.dta, clear

destring countyfips, replace force
destring stfips, replace force

sort countyfips modate

replace maxs = 0 if missing(maxs)

//----------------------------------------- constructing forward and backward lags
local LMAX     172
local FMAX     12

* Forward leads: _FF72_maxs, ..., _FF1_maxs
forvalues i = `FMAX'(-1)1 {
    quietly {
        gen _FF`i'_maxs = .
        bysort countyfips (modate): replace _FF`i'_maxs = maxs[_n + `i']
        replace _FF`i'_maxs = 0 if missing(_FF`i'_maxs)
    }
}

* Backward lags: _LL0_maxs, ..., _LL172_maxs
forvalues i = 0/`LMAX' {
    quietly {
        gen _LL`i'_maxs = .
        bysort countyfips (modate): replace _LL`i'_maxs = maxs[_n - `i']
        replace _LL`i'_maxs = 0 if missing(_LL`i'_maxs)
    }
}

//------------------------------------------------------------------------------
// Actual Nature lag exposure: months 0 through 172 after TC
//------------------------------------------------------------------------------

gen X_nature = 0

scalar nature_sum_lag = 0
forvalues l = 0/`LMAX' {
    scalar nature_sum_lag = nature_sum_lag + ///
        (0.0237 + 0.000535*`l' - 0.0000039*(`l'^2))
}

display "Nature actual lag-shape sum, 0-172 = " nature_sum_lag

forvalues l = 0/`LMAX' {
    scalar g_lag`l' = ///
        (0.0237 + 0.000535*`l' - 0.0000039*(`l'^2)) / nature_sum_lag

    replace X_nature = X_nature + g_lag`l' * _LL`l'_maxs
}

//------------------------------------------------------------------------------
// Compact future placebo exposure: months 1 through 72 before TC
// This tests whether future TC exposure predicts current mortality.
//------------------------------------------------------------------------------

gen X_future72 = 0

scalar nature_sum_future72 = 0
forvalues f = 1/`FMAX' {
    scalar nature_sum_future72 = nature_sum_future72 + ///
        (0.0237 + 0.000535*`f' - 0.0000039*(`f'^2))
}

display "Future placebo lag-shape sum, 1-72 = " nature_sum_future72

forvalues f = 1/`FMAX' {
    scalar g_future`f' = ///
        (0.0237 + 0.000535*`f' - 0.0000039*(`f'^2)) / nature_sum_future72

    replace X_future72 = X_future72 + g_future`f' * _FF`f'_maxs
}

save output/full_data_for_regression.dta, replace