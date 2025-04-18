//-------------------------------------------------------------DEFINE PROGRAM TO PREDICT TC DEATHS


program define attribute , nclass
args Y m fixedpop

use output/plotting_deaths.dta, clear

egen event_id = group(storm_serial_id cluster_label)
lab var event_id "unique state-by-storm identifier"

tsset event_id  modate 
tsfill, full

replace maxs = 0 if maxs == .

egen storm = group(year month storm_serial_id)
bysort event_id: egen storm_id = max(storm)
drop storm

encode storm_serial_id, gen(storm_name_id)
bysort event_id: egen x = max(storm_name_id)
label values x storm_name_id
decode x, gen(storm_name_2)
drop x
replace storm_serial_id = storm_name_2
drop storm_name_2

encode cluster_label, gen(adm_id)
bysort event_id: egen x = max(adm_id)
label values x adm_id
decode x, gen(cluster_label_2)
drop x
replace cluster_label = cluster_label_2
drop cluster_label_2

drop adm_id
drop storm_name_id
//drop year month



// --------------------------- MERGE IN POPULATION DATA

order cluster_label modate
sort cluster_label modate
merge m:1 cluster_label modate using output/population_data_for_plotting.dta
plot event_id modate if _merge == 3
keep if _merge == 3 | _merge == 1 // population data is not available prior to 1930
drop _merge

// --------------------------- MERGE IN BETA DATA

order cluster_label modate
sort cluster_label modate
if "`Y'"=="tdths_ttpop" {
merge m:1 cluster_label modate using output/beta/`m'_`Y'.dta
}
else {
merge m:1 cluster_label modate using output/beta/`m'_dths`Y'.dta	
}
plot event_id modate if _merge == 3

keep if _merge == 3 | _merge == 1 // since beta data is not available prior to 1950
drop _merge

if "`fixedpop'" == "4" { 
//--------------------------- CONSTRUCT PREDICTIONS WITH ACTUAL POPULATION

if "`Y'"=="tdths_ttpop" {
	di as err "Begin constructing predictions for `Y' `m'"
	tsset
	if "`m'" =="m_nonlinear" | "`m'" =="m_nonlinear_adapt_4" | "`m'" =="m_nonlinear_adapt_pool24" {
		gen mort = totalpop/100000*L0.maxs*beta_L0_maxs + totalpop/100000*L0.maxs^2*beta_L0_maxs_2

		forvalues i = 1/180 {
		qui {
			replace mort = mort + totalpop/100000*L`i'.maxs*beta_L`i'_maxs + totalpop/100000*L`i'.maxs_2*beta_L`i'_maxs_2
		}
		}
	}

	if "`m'" =="m_cubic_adapt_pool24" | "`m'" =="m_cubic" {
		gen mort = totalpop/100000*L0.maxs*beta_L0_maxs + totalpop/100000*L0.maxs^2*beta_L0_maxs_2 + totalpop/100000*L0.maxs^3*beta_L0_maxs_3

		forvalues i = 1/180 {
		qui {
			replace mort = mort + totalpop/100000*L`i'.maxs*beta_L`i'_maxs + totalpop/100000*L`i'.maxs_2*beta_L`i'_maxs_2 + totalpop/100000*L`i'.maxs_3*beta_L`i'_maxs_3
		}
		}
	}
	if "`m'" =="m_linear" |  "`m'" == "m_linear_leads" |  "`m'" == "m_linear_adapt_4" |  "`m'" == "m_linear_adapt_pool24" {
		gen mort = totalpop/100000*L0.maxs*beta_L0_maxs 
		forvalues i = 1/180 {
		qui {
			replace mort = mort + totalpop/100000*L`i'.maxs*beta_L`i'_maxs
		}
		}
	}
} // "`Y'"=="tdths_ttpop"

} // end of "`fixedpop'" == "4"

replace mort = 0 if mort<0 // Trucate the negatives
replace mort = . if modate < 817 // since have incomplete history of storms for dates before 1968

preserve 
drop beta_L*
save output/mortality_predict_`m'_`Y'.dta, replace //THIS MAKES THE month-state-storm numbers in the paper
restore

//------------Collapse by storm
preserve
di as err "collapsing mortality by serial id"

collapse (sum) mort, by(storm_serial_id  modate)

save output/mortality_predict_storm.dta, replace

restore

//------------Collapse by state
preserve
di as err "collapsing mortality by state"

//truncate at 0
replace mort = 0 if mort<0
collapse (sum) mort, by(cluster_label)

save output/mortality_predict_state.dta, replace
restore


//------------Collapse by date
preserve
di as err "collapsing mortality by date"

collapse (sum) mort, by(modate)

save output/mortality_predict_date.dta, replace
restore

end




//---------------------------------------------------------------------------------
//03-mortality-predictions
//---------------------------------------------------------------------------------

//------------- Create excess deaths data for Figure 4a (also need it for Table SI1)
local model_list "m_linear"
foreach n of loc model_list {
	attribute tdths_ttpop `n' 4
}


// convert to csv
use output/mortality_predict_m_linear_tdths_ttpop.dta, clear
export delimited using output/mortality_predict_m_linear_tdths_ttpop.csv, replace

use output/mortality_predict_storm.dta, clear
export delimited using output/mortality_predict_storm.csv, replace

use output/mortality_predict_state.dta, clear
export delimited using output/mortality_predict_state.csv, replace

use output/mortality_predict_date.dta, clear
export delimited using output/mortality_predict_date.csv, replace

