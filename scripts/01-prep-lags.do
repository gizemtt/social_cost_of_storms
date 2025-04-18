//------------------------------------------------------------------------------
//--------------------------------------------------- prepping temp and lag variables
//------------------------------------------------------------------------------

//monthly data during 1851-2015
use data/DATA_hurricane_mortality_temp_month_state_19302015_state.dta, clear

keep if inlist(stfips, 48, 22) // 🔸 Keep only Texas data

//----------------------------------------- constructing normal lags
forvalues i = 54(-1)1 {
	quietly{
	gen _FF`i'_maxs= F`i'.maxs
	}
}

forvalues i = 0/180 {
	quietly{
	gen _LL`i'_maxs= L`i'.maxs
	}
}

save output/full_data_for_regression.dta, replace /* THIS IS THE MAIN DATASET FOR ALL REGRESSIONS */

//-------------------------------------------------------------saving population data for merging into plotting data set

use "output/full_data_for_regression.dta", clear

keep month year adm_name totalpop

gen modate = (year-1900)*12 + month
lab var modate "months since jan 1900 = 1"
drop if year < 1953

replace adm_name = upper(adm_name)
sort adm_name modate

drop if adm_name==""

save "output/population_data_for_plotting.dta", replace



//---------------------------------------------------------computing and storm-specific mortality attribution

//-------------------------------------------------------------saving state quartile 

use "output/full_data_for_regression.dta", clear
egen maxs_mean_st = mean(maxs), by(stfips)

xtile pct = maxs_mean_st, n(4)
collapse (mean) pct, by(adm_name)
save output/pct_maxs.dta, replace


//monthly data during 1851-2015

insheet using "data/panel_by_storm__NA_USA_density_8_yr_1930_2018_state.csv", clear
keep if inlist(stfips, 48, 22) // 🔸 Keep only Texas data

gen modate = (year-1900)*12 + month
lab var modate "months since jan 1900 = 1"
drop if year < 1953 | year > 2015

rename adm_name adm_name_cap
gen adm_name = upper(adm_name_cap)
drop adm_name_cap

merge m:1 adm_name using output/pct_maxs.dta, keep(3) nogen //just keep the matches

gen firstdate_flag = .
egen min_date = min(modate)
sort firstdate_flag modate
bysort pct modate : gen seq=_n
replace firstdate_flag = 1 if modate==min_date
replace firstdate_flag = . if seq!=firstdate_flag

drop if maxs == 0 & firstdate_flag!=1 
drop firstdate_flag seq min_date

save output/plotting_deaths.dta, replace