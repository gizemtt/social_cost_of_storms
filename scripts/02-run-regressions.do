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

loc outcome_list "tdths_ttpop"
foreach Y of loc outcome_list {	
	// Run m_linear regression using reghdfe

	reghdfe tdths_ttpop _LL*_maxs, ///
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
	estimates save output/beta/m_linear_tdths_ttpop.ster, replace

	local model_list "m_linear"
	foreach m of loc model_list {
		
		//---- make beta for attribution
		preserve
		
		di "`m'_`Y'.ster"

		est use "output/beta/`m'_`Y'.ster"
		
		matrix list e(b)
		
		if "`m'"=="m_linear" | "`m'" == "m_nonlinear" | "`m'" == "m_linear_leads" | "`m'"=="m_cubic" {
			local z = 1
			gen pct_1 = 1
			
			di "model `z'"
		}
		
		keep cluster_label modate pct_* year month

		forvalues q=1/`z' {

			*** get coefficients from models
			*quietly {
			if `q'==1 {
				di "`q'"
				forvalues t=0/180 {
					
				gen beta_L`t'_maxs = .
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] if pct_`z'==`q'
				
				di "pct_`z'=`q'"
				
				if "`m'" == "m_nonlinear" | "`m'" == "m_nonlinear_adapt_4" | "`m'" == "m_nonlinear_adapt_pool24" {
				gen beta_L`t'_maxs_2 = .
				replace beta_L`t'_maxs_2=_b[_LL`t'_maxs_2] if pct_`z'==`q'
				}
				
				if "`m'" == "m_cubic" | "`m'" == "m_cubic_adapt_pool24" {
				gen beta_L`t'_maxs_2 = .
				replace beta_L`t'_maxs_2=_b[_LL`t'_maxs_2] if pct_`z'==`q'
				gen beta_L`t'_maxs_3 = .
				replace beta_L`t'_maxs_3=_b[_LL`t'_maxs_3] if pct_`z'==`q'
				}
				} /* end loop over t */
			} /* end q=1 */
			if `q'>1 {
				di "`q'>1"
				di "pct_`z'=`q'"

				forvalues t=0/180 {
				if "`m'"=="m_linear" | "`m'" == "m_nonlinear" | "`m'" == "m_linear_leads" {
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] + _b[`q'.pct_`z'#_LL`t'_maxs] if pct_`z'==`q' 
				}
				
				if "`m'"=="m_linear_adapt_4" | "`m'"=="m_linear_adapt_pool24" {
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] + _b[`q'.pct_`z'#c._LL`t'_maxs] if pct_`z'==`q' 				
				}
				
				if "`m'" == "m_nonlinear_adapt_4" {
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] + _b[`q'.pct_`z'#c._LL`t'_maxs] if pct_`z'==`q' 				
				replace beta_L`t'_maxs_2=_b[_LL`t'_maxs_2] + _b[`q'.pct_`z'#c._LL`t'_maxs_2] if pct_`z'==`q'
				}
				
				if "`m'" == "m_nonlinear" | "`m'" == "m_nonlinear_adapt_pool24" {
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] + _b[`q'.pct_`z'#_LL`t'_maxs] if pct_`z'==`q' 
				replace beta_L`t'_maxs_2=_b[_LL`t'_maxs_2] + _b[`q'.pct_`z'#_LL`t'_maxs_2] if pct_`z'==`q'
				}
				
				if "`m'" == "m_cubic" | "`m'" == "m_cubic_adapt_pool24" {
				replace beta_L`t'_maxs=_b[_LL`t'_maxs] + _b[`q'.pct_`z'#_LL`t'_maxs] if pct_`z'==`q' 
				replace beta_L`t'_maxs_2=_b[_LL`t'_maxs_2] + _b[`q'.pct_`z'#_LL`t'_maxs_2] if pct_`z'==`q'
				replace beta_L`t'_maxs_3=_b[_LL`t'_maxs_3] + _b[`q'.pct_`z'#_LL`t'_maxs_3] if pct_`z'==`q'
				}
				} /* end loop over t */
			} /* end else */ 
			*}
		}
		
		drop modate
		gen modate = (year-1900)*12 + month
		lab var modate "months since jan 1900 = 1"
		drop if year < 1953

		save "output/beta/`m'_`Y'.dta", replace

		restore

	}
	}