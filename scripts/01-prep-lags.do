//------------------------------------------------------------------------------
//--------------------------------------------------- prepping temp and lag variables
//------------------------------------------------------------------------------

//monthly data during 1948-2018
use data/DATA_hurricane_mortality_temp_month_county_19482018.dta, clear

destring countyfips, replace force
destring stfips, replace force

sort countyfips modate

//----------------------------------------- constructing normal lags
* Forward lags: _FF54_maxs, ..., _FF1_maxs
forvalues i = 9(-1)1 {
	quietly {
		gen _FF`i'_maxs = .
		bysort countyfips (modate): replace _FF`i'_maxs = maxs[_n + `i']
	}
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