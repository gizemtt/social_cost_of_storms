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

/* loc placebo_leads ""
forvalues i = 1/6 {
    loc placebo_leads "`placebo_leads' c._FF`i'_maxs#i.countyfips"
} */

* use cluster_code instead of countyfips if you would like to use cluster of counties for that term!
* Define interaction terms
local placebo_leads ""
foreach m of local placebo_months {
    local placebo_leads "`placebo_leads' c._FF`m'_maxs#i.countyfips"
}

* export delimited using "output/full_data_for_regression2.csv", replace

//--------------------------------------------------Run regressions
ssc install reghdfe, replace
ssc install ftools, replace

* use cluster_code instead of countyfips if you would like to use cluster of counties for that term!
reghdfe tdths_ttpop c.X0#i.countyfips c.X1#i.countyfips c.X2#i.countyfips `placebo_leads', ///
	absorb( ///
		i.countyfips#i.month ///
		i.countyfips#c.modate_1 i.countyfips#c.modate_2 ///
		i.countyfips#c.modate_3 i.countyfips#c.modate_4 ///
		i.countyfips#c.modate_5 i.countyfips#c.modate_6 ///
		i.countyfips#c.modate_7 i.countyfips#c.modate_8 ///
		i._im_#c.modate_1 ///
		i.countyfips#c.tavg i.countyfips#c.tavg_2 ///
		i.modate ///
	) ///
	vce(cluster countyfips) tol(1e-3) resid
estimates save output/beta/m_quadpoly_tdths_ttpop.ster, replace

* AIC/BIC compare in-sample fit while penalizing complexity. Choose the lower AIC model.
* AIC penalizes model complexity (via the number of parameters df) while rewarding goodness of fit.
* BIC is more conservative — better suited when you care more about parsimony and potential out-of-sample generalizability.
estat ic

* Predict fitted values and residuals
predict double resid, residuals
predict double fitted, xb

* Diagnostics
gen sq_resid = resid^2
gen sq_fitted = fitted^2

summarize resid fitted
summarize sq_resid sq_fitted

display "Adjusted R-squared: " e(r2_a)

* Export for diagnostics in Python
keep countyfips year month resid fitted
save "output/beta/residuals_fitted.dta", replace
export delimited using "output/beta/residuals_fitted.csv", replace