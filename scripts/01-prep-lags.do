//------------------------------------------------------------------------------
//--------------------------------------------------- prepping temp and lag variables
//------------------------------------------------------------------------------

//monthly data during 1851-2015
use data/DATA_hurricane_mortality_temp_month_county_19482018.dta, clear

destring countyfips, replace force
destring stfips, replace force

sort countyfips modate

keep if inlist(stfips, 48, 22) // 🔸 Keep only Texas data

gen cluster_label = "CLUSTER_" + string(countyfips, "%02.0f")

//----------------------------------------- constructing normal lags
* Forward lags: _FF54_maxs, ..., _FF1_maxs
forvalues i = 54(-1)1 {
	quietly {
		gen _FF`i'_maxs = .
		bysort countyfips (modate): replace _FF`i'_maxs = maxs[_n + `i']
	}
}

* Backward lags: _LL0_maxs, ..., _LL180_maxs
forvalues i = 0/180 {
	quietly {
		gen _LL`i'_maxs = .
		bysort countyfips (modate): replace _LL`i'_maxs = maxs[_n - `i']
	}
}

save output/full_data_for_regression.dta, replace /* THIS IS THE MAIN DATASET FOR ALL REGRESSIONS */

//-------------------------------------------------------------saving population data for merging into plotting data set

use "output/full_data_for_regression.dta", clear

keep month year cluster_label totalpop

gen modate = (year-1900)*12 + month
lab var modate "months since jan 1900 = 1"
drop if year < 1953

replace cluster_label = upper(cluster_label)
sort cluster_label modate

drop if cluster_label==""

save "output/population_data_for_plotting.dta", replace



//---------------------------------------------------------computing and storm-specific mortality attribution

//-------------------------------------------------------------saving state quartile 

use "output/full_data_for_regression.dta", clear
egen maxs_mean_st = mean(maxs), by(countyfips)

xtile pct = maxs_mean_st, n(4)
collapse (mean) pct, by(cluster_label)
save output/pct_maxs.dta, replace


//monthly data during 1851-2015

insheet using "data/panel_by_storm__NA_USA_density_8_yr_1930_2018.csv", clear
destring countyfips, replace force
destring stfips, replace force
keep if inlist(stfips, 48, 22) // 🔸 Keep only Texas data

rename stormid storm_serial_id
rename stormname storm_name

gen modate = (year-1900)*12 + month
lab var modate "months since jan 1900 = 1"
drop if year < 1953 | year > 2015

gen cluster_label = "CLUSTER_" + string(countyfips, "%02.0f")

rename cluster_label cluster_label_cap
gen cluster_label = upper(cluster_label_cap)
drop cluster_label_cap

merge m:1 cluster_label using output/pct_maxs.dta, keep(3) nogen //just keep the matches

gen firstdate_flag = .
egen min_date = min(modate)
sort firstdate_flag modate
bysort pct modate : gen seq=_n
replace firstdate_flag = 1 if modate==min_date
replace firstdate_flag = . if seq!=firstdate_flag

drop if maxs == 0 & firstdate_flag!=1 
drop firstdate_flag seq min_date

save output/plotting_deaths.dta, replace