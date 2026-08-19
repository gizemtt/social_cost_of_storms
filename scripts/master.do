cd "C:\Users\toplu\Documents\GithubRepositories\social_cost_of_storms"
*set maxvar 32767

timer clear

// Start total timer
timer on 1

// 01 - PREP
timer on 2
do "scripts/01-prep-lags.do"
timer off 2
timer list 2

// 02 - REGRESSION
timer on 3
do "scripts/02-run-regressions.do"
timer off 3
timer list 3

// 03 - postprocess
timer on 4
do "scripts\03-postprocess-coefficients.do"
timer off 4
timer list 4

// Stop total timer
timer off 1

// Final timing summary
di "----------------------------------"
di "   ⏱️ Timing Summary (seconds)"
di "----------------------------------"
foreach i in 1 2 3 4 {
    timer list `i'
}
