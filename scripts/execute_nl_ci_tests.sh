#!/bin/bash

cmd="python benchmark.py -m method="
cmd2="ds_name="
cmd3="which_dataset='range(0,40)'  data_base_path="/path_to/data_release/" save_path='/path_to/results_nl_ci/'"


data_paths=(
    "nl_rbf_small"
    "nl_comp_small"

)

# We use this to run pcmci and pcmci plus with nonlinear ci_tests on the nl datasets. 
#We use a different script for this, since we need to set a different save path and data base path.
methods=(
"pcmci ci_test=GPDC"
"pcmciplus ci_test=GPDC method.reset_lagged_links=True,False "
)



for data_path in "${data_paths[@]}"; do
    echo "$data_path"
    for method in "${methods[@]}"; do
        if [[ "$data_path" == *"big"* ]]; then
            cmd4=" method.max_lag=4" 
        fi 
        if [[ "$data_path" == *"small"* ]]; then
            cmd4=" method.max_lag=3" 
        fi

        echo  "$cmd$method $cmd2$data_path $cmd3$cmd4" 
        eval "$cmd$method $cmd2$data_path $cmd3$cmd4" 



    done
done

wait
echo "Done"