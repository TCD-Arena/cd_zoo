#!/bin/bash

# This runs experiments for a single dataset of an assumption violation.
# For the sets with 7 variables, a bigger max lag is used.
# Usage: ./execute_all_methods.sh <data_path1> [data_path2] ... [--method <method_name>]
# If --method is specified, only that method will be run. Otherwise, all methods are run.

# Parse arguments
data_paths=()
specific_method=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            specific_method="$2"
            shift 2
            ;;
        *)
            data_paths+=("$1")
            shift
            ;;
    esac
done

# If no data paths provided, show usage
if [[ ${#data_paths[@]} -eq 0 ]]; then
    echo "Usage: $0 <data_path1> [data_path2] ... [--method <method_name>]"
    echo "Available methods:"
    echo "  - direct_crosscorr"
    echo "  - varlingam" 
    echo "  - var"
    echo "  - pcmci"
    echo "  - pcmciplus"
    echo "  - dynotears"
    echo "  - ntsnotears"
    echo "  - cp"
    exit 1
fi

cd ..

cmd="python benchmark.py -m method="
cmd2="ds_name="
cmd3="which_dataset='range(0,40)'"


# This is the full Hyperparameter space the we use for throughout our experiments.
methods=(
"direct_crosscorr"
"varlingam method.prune=True,False"
"var method.base_on=coefficients,p_values "
"pcmci ci_test=ParCorr,RobustParCorr"
"pcmciplus ci_test=ParCorr,RobustParCorr method.reset_lagged_links=True,False "
"dynotears method.lambda_w=0.1,0.3 method.lambda_a=0.1,0.3 method.max_iter=100,40 method.h_tol=1e-8,1e-5"
"ntsnotears method.h_tol=1e-60,1e-10 method.rho_max=1e+16,1e+18 method.lambda1=0.005,0.001 method.lambda2=0.01,0.001 restrict_to_n_samples=33"
"cp method.architecture=transformer,unidirectional"
"svarrfci ci_test=ParCorr,RobustParCorr"
"fpcmci method.ci_test=parcorr,robust_parcorr" # Note, we use a different environment for this method.
)

# Filter methods if specific method is requested
if [[ -n "$specific_method" ]]; then
    filtered_methods=()
    for method in "${methods[@]}"; do
        if [[ "$method" == "$specific_method"* ]]; then
            filtered_methods+=("$method")
        fi
    done
    if [[ ${#filtered_methods[@]} -eq 0 ]]; then
        echo "Error: Method '$specific_method' not found."
        echo "Available methods: crosscorr, varlingam, var, pcmci, pcmciplus, dynotears, ntsnotears, cp, lpcmci svarrfci"
        exit 1
    fi
    
    methods=()
    for m in "${filtered_methods[@]}"; do
        method_name="${m%% *}"
        if [[ "$method_name" == "$specific_method" ]]; then
            methods+=("$m")
        fi
    done

    if [[ ${#methods[@]} -eq 0 ]]; then
        echo "Error: Method '$specific_method' not found."
        echo "Available methods: crosscorr, varlingam, var, pcmci, pcmciplus, dynotears, ntsnotears, cp, lpcmci svarrfci"
        exit 1
    fi
    echo "Running specific method: $specific_method"
else
    echo "Running all methods"
fi


for data_path in "${data_paths[@]}"; do
    echo "$data_path"
    for method in "${methods[@]}"; do
        if [[ "$data_path" == *"big"* ]]; then
            cmd4=" method.max_lag=2,4,6" 
        fi 
        if [[ "$data_path" == *"small"* ]]; then
            cmd4=" method.max_lag=1,3,5" 
        fi
        if [[ "$data_path" == *"tiny"* ]]; then
            cmd4=" method.max_lag=1,2,3" 
        fi
        if [[ "$data_path" == *"large"* ]]; then
            cmd4=" method.max_lag=3" 
        fi


        if [[ "$method" == *"physical "* ||  "$method" == *"cp"* ]]; then
            echo "$cmd$method $cmd2$data_path $cmd3"
            eval "$cmd$method $cmd2$data_path $cmd3" 
        else
            echo  "$cmd$method $cmd2$data_path $cmd3$cmd4" 
            eval "$cmd$method $cmd2$data_path $cmd3$cmd4" 
        fi



    done
done

wait
echo "Done"