//------------------------------------------------------------------------------
//--------------------------------------------------- prepping temp and lag variables
//------------------------------------------------------------------------------

//monthly data during 1948-2018
use data/DATA_hurricane_mortality_temp_month_county_19482018.dta, clear

destring countyfips, replace force
destring stfips, replace force

xtset countyfips modate
sort countyfips modate

* keep if inlist(stfips, 48, 22) // 🔸 Keep only Texas data

gen cluster_label = "CLUSTER_" + string(countyfips, "%02.0f")

//----------------------------------------- constructing normal lags
* Forward lags: _FF54_maxs, ..., _FF1_maxs
/* forvalues i = 54(-1)1 {
	quietly {
		gen _FF`i'_maxs = .
		bysort countyfips (modate): replace _FF`i'_maxs = maxs[_n + `i']
	}
} */

* Define arbitrary placebo months
local placebo_months 5 10 15 20 25

foreach m of local placebo_months {
    gen _FF`m'_maxs = F`m'.maxs
}

* Backward lags: _LL0_maxs, ..., _LL180_maxs
forvalues i = 0/172 {
	quietly {
		gen _LL`i'_maxs = .
		bysort countyfips (modate): replace _LL`i'_maxs = maxs[_n - `i']
	}
}

gen X0 = 0
gen X1 = 0
gen X2 = 0

forvalues l = 0/172 {
    replace X0 = X0 + _LL`l'_maxs
    replace X1 = X1 + `l' * _LL`l'_maxs
    replace X2 = X2 + `= `l'^2' * _LL`l'_maxs
}

save output/full_data_for_regression.dta, replace /* THIS IS THE MAIN DATASET FOR ALL REGRESSIONS */