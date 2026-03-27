#!/bin/bash

# Usage: ./run_all_methods_on_data_release.sh [--method <method_name>] [filter_string] [limit_n]
#   --method <method_name>: only run the specified method (optional)
#   filter_string: only process folders containing this string (optional)
#   limit_n: only process the first n matching folders (optional)

# Parse arguments
specific_method=""
filter_string=""
limit_n=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            specific_method="$2"
            shift 2
            ;;
        *)
            if [[ -z "$filter_string" ]]; then
                filter_string="$1"
            elif [[ -z "$limit_n" ]]; then
                limit_n="$1"
            fi
            shift
            ;;
    esac
done

# Methods and their hyperparameters (copied from execute_all_methods.sh)
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
"fpcmci method.ci_test=parcorr,robust_parcorr"
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
        echo "Available methods: direct_crosscorr, varlingam, var, pcmci, pcmciplus, dynotears, ntsnotears, cp, svarrfci, fpcmci"
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
        echo "Available methods: direct_crosscorr, varlingam, var, pcmci, pcmciplus, dynotears, ntsnotears, cp, svarrfci, fpcmci"
        exit 1
    fi
    echo "Running specific method: $specific_method"
else
    echo "Running all methods"
fi

cmd="python cd_zoo/benchmark.py -m data_base_path=data_release/ method="
cmd2="ds_name="
cmd3="which_dataset='range(0,40)'"

count=0
for data_path in data_release/*/; do
    if [[ ! -d "$data_path" ]]; then
        continue
    fi
    folder_name=$(basename "$data_path")

    # If filter_string is set, skip folders that don't match
    if [[ -n "$filter_string" && "$folder_name" != *"$filter_string"* ]]; then
        continue
    fi

    # If limit_n is set, stop after n matches
    if [[ -n "$limit_n" && $count -ge $limit_n ]]; then
        break
    fi

    for method in "${methods[@]}"; do
        cmd4=""
        if [[ "$folder_name" == *"big"* ]]; then
            cmd4=" method.max_lag=2,4,6"
        elif [[ "$folder_name" == *"small"* ]]; then
            cmd4=" method.max_lag=1,3,5"
        elif [[ "$folder_name" == *"tiny"* ]]; then
            cmd4=" method.max_lag=1,2,3"
        elif [[ "$folder_name" == *"large"* ]]; then
            cmd4=" method.max_lag=3"
        fi

        if [[ "$method" == *"physical "* ||  "$method" == *"cp"* ]]; then
            echo "$cmd$method $cmd2$folder_name $cmd3"
            eval "$cmd$method $cmd2$folder_name $cmd3"
        else
            echo "$cmd$method $cmd2$folder_name $cmd3$cmd4"
            eval "$cmd$method $cmd2$folder_name $cmd3$cmd4"
        fi
    done
    count=$((count+1))
done

echo "Done"