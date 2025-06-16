* cd "C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms\state_beta_l_i_incomplete"

//---------------------------------------------------------------------------------
// 03-postprocess-coefficients.do
// Extract DLM coefficients and variance-covariance matrix
//---------------------------------------------------------------------------------

* Clean up memory
capture estimates drop _all
clear

* Step 1: Load model estimates
estimates use "output/beta/m_quadpoly_tdths_ttpop.ster"

*-----------------------------
* Step 2: Export variance-covariance matrix
*-----------------------------

matrix V = e(V)
local colnames : colnames V

* Ensure overwrite
cap erase "output/beta/vcv_region_poly.csv"

mata:
    // Retrieve VCV matrix and column names
    V = st_matrix("V")
    colnames = tokens("`colnames'")
    n = cols(V)

    // Open CSV file for writing
    fh = fopen("output/beta/vcv_region_poly.csv", "w")

    // Write header row
    header = "rowname"
    for (i = 1; i <= n; i++) {
        header = header + "," + colnames[i]
    }
    fput(fh, header)

    // Write each row of the matrix
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

* Clear current dataset to avoid conflicts
clear

* Ensure parmest is available
cap which parmest
if _rc {
    ssc install parmest
}

* Extract model parameters
parmest, norestore
gen is_placebo = strpos(parm, "_FF") > 0

* Save as Stata and CSV for further analysis
save "output/beta/coefficients_region_poly.dta", replace
export delimited using "output/beta/coefficients_region_poly.csv", replace