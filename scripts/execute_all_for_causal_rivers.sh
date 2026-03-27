#!/bin/bash


# Runs methods for Causal Rivers. Use the best configuration for each method, which is determined based on the results from the main benchmark experiments.

cmd="python run_methods_on_causal_rivers.py  method="

methods=(
"direct_crosscorr"
"varlingam method.prune=True"
"var method.absolute_coefficients=False"
"pcmci ci_test=ParCorr"
"pcmciplus ci_test=ParCorr method.reset_lagged_links=False"
"dynotears method.lambda_w=0.1 method.lambda_a=0.1 method.max_iter=100 method.h_tol=1e-8"
"ntsnotears method.h_tol=1e-60 method.rho_max=1e+16 method.lambda1=0.005 method.lambda2=0.01"
)


 cmd4=" method.max_lag=3" 

for method in "${methods[@]}"; do
        echo  "$cmd$method$cmd4" 
        eval "$cmd$method$cmd4" 
done

wait
echo "Done"