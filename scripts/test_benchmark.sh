#!/bin/bash

# Test script to run benchmark.py with all available causal discovery methods
# This script runs each method sequentially on the benchmark datasets

cd ..

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Benchmarking All Causal Discovery Methods"
echo "======================================"
echo ""

# Array of all available methods from method_loader.py
METHODS=(
    "direct_crosscorr"
    "var"
    "varlingam"
    "pcmci"
    "pcmciplus"
    "dynotears"
    "ntsnotears"
    "svarrfci"
    "cp"
    "fpcmci"
    "physical"
    "combo"
    "crosscorr"
    "svarfci"
    "cdmi"
)

# Check if a specific dataset folder is provided as argument
if [ $# -eq 0 ]; then
    echo -e "${BLUE}No dataset specified. Using default from config.${NC}"
    DATASET_ARG=""
else
    DATASET_FOLDER=$1
    echo -e "${BLUE}Using dataset folder: $DATASET_FOLDER${NC}"
    DATASET_ARG="data_path=sample_datasets/$DATASET_FOLDER"
fi

# Check if dataset index is provided
if [ $# -ge 2 ]; then
    DATASET_INDEX=$2
    echo -e "${BLUE}Using dataset index: $DATASET_INDEX${NC}"
    INDEX_ARG="which_dataset=$DATASET_INDEX"
else
    INDEX_ARG=""
fi

echo ""

# Counters for results
TOTAL=${#METHODS[@]}
SUCCESS=0
FAILED=0

# Results array
declare -a RESULTS

echo "Total methods to benchmark: $TOTAL"
echo ""

# Loop through all methods
for METHOD in "${METHODS[@]}"; do
    echo "--------------------------------------"
    echo -e "${YELLOW}Benchmarking method: $METHOD${NC}"
    echo "--------------------------------------"
    
    # Build the command with optional arguments
    CMD="python benchmark.py per_sample_metrics=False restrict_to_n_samples=3 method=\"$METHOD\" save=False"
    [ -n "$DATASET_ARG" ] && CMD="$CMD $DATASET_ARG"
    [ -n "$INDEX_ARG" ] && CMD="$CMD $INDEX_ARG"
    
    # Run the method with hydra override and show output directly
    if eval $CMD 2>&1; then
        echo -e "${GREEN}✓ SUCCESS${NC}: $METHOD"
        RESULTS+=("SUCCESS: $METHOD")
        ((SUCCESS++))
    else
        EXIT_CODE=$?
        echo -e "${RED}✗ FAILED${NC}: $METHOD (exit code: $EXIT_CODE)"
        RESULTS+=("FAILED: $METHOD (exit code: $EXIT_CODE)")
        ((FAILED++))
    fi
    
    echo ""
done

# Print summary
echo "======================================"
echo "BENCHMARK SUMMARY"
echo "======================================"
echo "Total methods benchmarked: $TOTAL"
echo -e "${GREEN}Successful: $SUCCESS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

echo "Detailed results:"
for RESULT in "${RESULTS[@]}"; do
    if [[ $RESULT == SUCCESS* ]]; then
        echo -e "  ${GREEN}$RESULT${NC}"
    else
        echo -e "  ${RED}$RESULT${NC}"
    fi
done

echo ""
echo "======================================"

# Exit with error if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
