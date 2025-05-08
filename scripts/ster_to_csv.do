capture estimates drop _all

* Load the model
* cd "C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms\90percent-statetimeimpacts"
estimates use output/beta/m_linear_tdths_ttpop.ster

* Extract the variance-covariance matrix
matrix V = e(V)

* Drop the _cons row and column (assumed to be last)
matrix V = V[1..rownumb(V, "_cons")-1, 1..colnumb(V, "_cons")-1]

* Assign row and column names: beta_L{lag}_maxs
local names
forvalues lag = 0/180 {
    local names `names' beta_L`lag'_maxs
}
matrix rownames V = `names'
matrix colnames V = `names'

* Save matrix to variables
clear
svmat V, names(col)

* Export as CSV
export delimited using "vcv.csv", replace

