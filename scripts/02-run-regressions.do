//---------------------------------------------------------------------------------
//02-run-regressions.do
//---------------------------------------------------------------------------------

//---------------------------------------------- creating fixed effects

use output/full_data_for_regression.dta, clear

* Merge all obs; keep match status
merge m:1 countyfips using "data/national_clustering.dta"

/* * Inspect merge results
tab _merge
* Optional: check unmatched counties
list countyfips if _merge != 3
* Optional: label _merge variable for clarity
label define mrg 1 "master only" 2 "using only" 3 "matched"
label values _merge mrg
* Export as CSV
export delimited using "output/full_data_for_regression2.csv", replace */

*centering modate on sample
sum modate if year >= 1968
replace modate = modate - round(r(mean))

forvalues i = 1/8 {
	gen modate_`i' = modate^`i'
}

*temperature nonlinearity
gen tavg_2 = tavg^2

egen _im_ = group(countyfips month)

//--------------------------------------------------Run regressions
ssc install reghdfe, replace
ssc install ftools, replace

reghdfe tdths_ttpop c.X0 c.X1 c.X2##i.cluster_code, ///
	absorb( ///
		i.countyfips#i.month ///
		i.countyfips#c.modate_1 i.countyfips#c.modate_2 ///
		i.countyfips#c.modate_3 i.countyfips#c.modate_4 ///
		i.countyfips#c.modate_5 i.countyfips#c.modate_6 ///
		i.countyfips#c.modate_7 i.countyfips#c.modate_8 ///
		i._im_#c.modate_1 ///
		i.countyfips#c.tavg i.countyfips#c.tavg_2 ///
		i.modate ///
	)
vce(cluster countyfips) // or your preferred level
estimates save output/beta/m_quadpoly_tdths_ttpop.ster, replace