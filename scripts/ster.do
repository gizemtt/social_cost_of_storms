capture estimates drop _all

* Step 1: Load the model and extract V
* cd "C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms\state_beta_l_i_incomplete"
estimates use output/beta/m_linear_tdths_ttpop.ster

matrix V = e(V)
local colnames : colnames V

* Step 2: Export via Mata — row/column named CSV
mata:
    // Pull matrix and names from Stata
    V = st_matrix("V")
    colnames = tokens("`colnames'")
    n = cols(V)

    // Open CSV file for writing
    fh = fopen("vcv_beta_li_full.csv", "w")

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

    // Close file
    fclose(fh)
end



* === Export Coefficients with parmest === *

* Drop current data to allow parmest output
clear

* Ensure parmest is installed
cap which parmest
if _rc {
    ssc install parmest
}

* Run parmest (extracts all coef + stats)
parmest, norestore

* Save output to .dta file for Python postprocessing
save "beta_li_output.dta", replace

