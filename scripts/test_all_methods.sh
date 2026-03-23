#!/bin/bash

# Test script to run example_run.py with all available causal discovery methods
# This script runs each method sequentially and logs the results

cd ..


# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Testing All Causal Discovery Methods"
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
    "mean"
    "combo"
    "crosscorr"
    "svarfci"
    "cdmi"
)

# Counters for results
TOTAL=${#METHODS[@]}
SUCCESS=0
FAILED=0
SKIPPED=0

# Results array
declare -a RESULTS

echo "Total methods to test: $TOTAL"
echo ""

# Loop through all methods
for METHOD in "${METHODS[@]}"; do
    echo "--------------------------------------"
    echo -e "${YELLOW}Testing method: $METHOD${NC}"
    echo "--------------------------------------"
    
    # Run the method with hydra override and show output directly
    if python example_run.py method="$METHOD" 2>&1; then
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
echo "TEST SUMMARY"
echo "======================================"
echo "Total methods tested: $TOTAL"
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
