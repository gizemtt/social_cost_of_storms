//---------------------------------------------------------------------------------
// 03-postprocess-coefficients.do
// Extract model coefficients and variance-covariance matrix
//---------------------------------------------------------------------------------

* Clean up memory
capture estimates drop _all
clear

*-----------------------------
* Step 1: Load model estimates
*-----------------------------
estimates use "output/beta/m_quadpoly_tdths_ttpop.ster"

*-----------------------------
* Step 2: Export variance-covariance matrix
*-----------------------------

matrix V = e(V)
local colnames : colnames V

cap erase "output/beta/vcv_region_poly.csv"

mata:
    V = st_matrix("V")
    colnames = tokens("`colnames'")
    n = cols(V)

    fh = fopen("output/beta/vcv_region_poly.csv", "w")

    header = "rowname"
    for (i = 1; i <= n; i++) {
        header = header + "," + colnames[i]
    }
    fput(fh, header)

    for (i = 1; i <= rows(V); i++) {
        row = colnames[i]
        for (j = 1; j <= cols(V); j++) {
            row = row + "," + strofreal(V[i,j])
        }
        fput(fh, row)
    }

    fclose(fh)
end

*-----------------------------
* Step 3: Extract coefficients via parmest
*-----------------------------

cap which parmest
if _rc {
    ssc install parmest
}

parmest, norestore

*-----------------------------
* Step 4: Classify coefficient types
*-----------------------------

gen term_type = ""

replace term_type = "actual_county_nature" if strpos(parm, "X_nature") > 0
replace term_type = "future_placebo"      if strpos(parm, "X_future72") > 0
replace term_type = "constant"            if parm == "_cons"

gen is_actual  = term_type == "actual_county_nature"
gen is_placebo = term_type == "future_placebo"

* Extract countyfips for county-specific X_nature coefficients
gen countyfips = ""

replace countyfips = regexs(1) if regexm(parm, "^([0-9]+)\.countyfips#c\.X_nature")

* Pooled placebo has no countyfips
replace countyfips = "" if is_placebo

*-----------------------------
* Step 5: Save coefficient table
*-----------------------------

save "output/beta/coefficients_county_nature_future.dta", replace
export delimited using "output/beta/coefficients_region_poly.csv", replace