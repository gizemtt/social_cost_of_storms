//---------------------------------------------------------------------------------
//02-run-regressions.do
//---------------------------------------------------------------------------------

//---------------------------------------------- creating fixed effects

use output/full_data_for_regression.dta, clear

* Merge all obs; keep match status
* merge m:1 countyfips using "data/national_clustering.dta"
* rename _merge _merge_weights
* merge m:1 countyfips year month using "output/beta/fgls_weights.dta"


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
gen tavg_3 = tavg^3
egen _im_ = group(countyfips month)

loc placebo_leads ""
forvalues i = 1/9 {
    loc placebo_leads "`placebo_leads' c._FF`i'_maxs#i.countyfips"
}

* use cluster_code instead of countyfips if you would like to use cluster of counties for that term!

* export delimited using "output/full_data_for_regression2.csv", replace

//--------------------------------------------------Run regressions
* ssc install reghdfe, replace
* ssc install ftools, replace

* use cluster_code instead of countyfips if you would like to use cluster of counties for that term! [aw=w_fgls] [aw=totalpop] dths_tot tdths_ttpop . reghdfe tdths_ttpop ibn.countyfips#c.X0 ibn.countyfips#c.X1 ibn.countyfips#c.X2 [aw=totalpop], ///

//-------------------------------------------------- Run regression. ibn.countyfips#c.X_nature  ibn.svi_high#c.X_nature ///    ibn.svi_q#c.X_future72 ///  replace svi_high = 1 if inlist(svi_q, 2, 3)
bysort countyfips: egen mean_svi_county = mean(svi)

preserve
    keep countyfips mean_svi_county
    duplicates drop

    gen svi_high = .
    replace svi_high = 0 if mean_svi_county <= 0.75
    replace svi_high = 1 if mean_svi_county > 0.75 & mean_svi_county < .

    tempfile svi_groups
    save `svi_groups'
restore

merge m:1 countyfips using `svi_groups', nogen

reghdfe tdths_ttpop ///
    ibn.svi_high#c.X_nature ///
    ibn.svi_high#c.X_future72 ///
    [aw=totalpop], ///
    absorb( ///
        i.countyfips#i.month ///
        i.countyfips#c.modate_1 ///
        i._im_#c.modate_1 ///
        i.countyfips#c.tavg i.countyfips#c.tavg_2 ///
        i.modate ///
    ) ///
    vce(cluster countyfips) tol(1e-3) resid

//--------------- Coefficient Tests
* Two-sided test (This asks whether the TC exposure coefficient for the specific group counties is statistically different from zero in either direction. 
* Stata reports an F-test(wald), but because this is a single coefficient restriction (degree of freedom=1), it is equivalent to a two-sided t-test (F=t^2).)
* null: beta = 0 against H_A: beta not zero.
* If we fail to reject the null hypothesis (p>0.05), this does not prove the effect is zero. It means the data do not provide enough statistical evidence that 
* this coefficient differs from zero.
test 0.svi_high#c.X_nature = 0
test 1.svi_high#c.X_nature = 0

* One-sided test (one sided t-test) (the most important)
* This asks whether there is evidence of a positive TC mortality response among cluster counties. 
* H_0: beta_k <= 0 against H_A: beta_k > 0
lincom 1.svi_high#c.X_nature
display "One-sided p-value for high SVI TC effect > 0 = " ttail(r(df), r(estimate)/r(se))
lincom 0.svi_high#c.X_nature
display "One-sided p-value for low SVI TC effect > 0 = " ttail(r(df), r(estimate)/r(se))

* Joint test (joint F-test (Wald)) : no need to present 
* This asks whether TC exposure has any detectable mortality association in either SVI group. 
* It does not specifically test whether high-SVI counties have larger effects than low-SVI counties.
* null: beta_low=0 and beta_high=0 against at least one is non-zero.
testparm ibn.svi_high#c.X_nature

//--------------- Testing whether high-SVI effects are larger than low-SVI effects

* Two-sided test: high-SVI effect equals low-SVI effect
* Null: beta_high - beta_low = 0 against beta_high - beta_low =! 0
test 1.svi_high#c.X_nature = 0.svi_high#c.X_nature

* One-sided test: high-SVI effect is larger than low-SVI effect
* Null: beta_high - beta_low = 0 against beta_high - beta_low > 0
lincom 1.svi_high#c.X_nature - 0.svi_high#c.X_nature
display "One-sided p-value for high-SVI effect > low-SVI effect = " ///
    ttail(r(df), r(estimate)/r(se))


//--------------- Placebo lead tests
* Placebo Two-sided test (This asks whether the TC exposure coefficient for the specific group counties is statistically different from zero in either direction. 
* Stata reports an F-test(wald), but because this is a single coefficient restriction (degree of freedom=1), it is equivalent to a two-sided t-test.)
* null: beta = 0 against H_A: beta not zero.
* If we fail to reject the null hypothesis (p>0.05), this does not prove the effect is zero. It means the data do not provide enough statistical evidence that 
* this coefficient differs from zero.
test 0.svi_high#c.X_future72 = 0
test 1.svi_high#c.X_future72 = 0

* Placebo One-sided test (one sided t-test) (the most important)
* This asks whether there is evidence of a positive TC mortality response among cluster counties. 
* H_0: beta_k = 0 against H_A: beta_k > 0
lincom 1.svi_high#c.X_future72
display "One-sided p-value for high SVI placebo > 0 = " ttail(r(df), r(estimate)/r(se))
lincom 0.svi_high#c.X_future72
display "One-sided p-value for low SVI placebo > 0 = " ttail(r(df), r(estimate)/r(se))

* Joint placebo test (F-test)
* Future TC exposure should not predict current mortality. If this test is significant, that suggests pre-trends, confounding, or model misspecification.
testparm ibn.svi_high#c.X_future72

//-------------------------------------------------- Save model
estimates save output/beta/m_quadpoly_tdths_ttpop.ster, replace

* AIC/BIC compare in-sample fit while penalizing complexity. Choose the lower AIC model.
* AIC penalizes model complexity (via the number of parameters df) while rewarding goodness of fit.
* BIC is more conservative — better suited when you care more about parsimony and potential out-of-sample generalizability.
estat ic

//-------------------------------------------------- Predict fitted values and residuals . capture drop resid fitted fitted_full sq_resid sq_fitted

predict double resid, residuals
predict double fitted, xb
predict double fitted_full, xbd

gen sq_resid = resid^2
gen sq_fitted = fitted^2

summarize resid fitted
summarize sq_resid sq_fitted

display "Adjusted R-squared: " e(r2_a)

//-------------------------------------------------- Export residual diagnostics data

keep countyfips year month resid fitted fitted_full
save "output/beta/residuals_fitted.dta", replace
export delimited using "output/beta/residuals_fitted.csv", replace