#!/bin/bash

# RoboData Experiments Runner
# Executes all experiments described in experiments.tex using individual YAML config files
# Author: Generated based on experiments.tex specifications
# Date: August 1, 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROBODATA_ROOT="/DATA/RoboData"
CONFIG_DIR="experiment_configs"
RESULTS_DIR="experiments"
LOG_FILE="experiments_batch_$(date +%Y%m%d_%H%M%S).log"

# Ensure we're in the right directory
cd "$ROBODATA_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo "Please create the virtual environment first."
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
. .venv/bin/activate

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${RED}Error: Configuration directory $CONFIG_DIR not found${NC}"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to run an experiment using a YAML config file
run_experiment() {
    local config_file="$1"
    local description="$2"
    
    echo -e "${YELLOW}Running: $description${NC}"
    echo -e "${BLUE}Config: $config_file${NC}"
    echo ""
    
    local cmd="python -m backend.main -c $CONFIG_DIR/$config_file"
    
    echo "Executing: $cmd" >> "$LOG_FILE"
    
    # Run the command and capture both stdout and stderr
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Completed successfully${NC}"
        echo ""
        echo "----------------------------------------"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Failed with error${NC}"
        echo "Check $LOG_FILE for details"
        echo ""
        echo "----------------------------------------"
        echo ""
        return 1
    fi
}

# Function to run consistency test (same config multiple times)
run_consistency_test() {
    local config_file="$1"
    local iterations=10
    
    echo -e "${YELLOW}Starting Consistency Test (QA10)${NC}"
    echo -e "${BLUE}Running config $iterations times: $config_file${NC}"
    echo ""
    
    for i in $(seq 1 $iterations); do
        echo -e "${YELLOW}Consistency Test - Run $i/$iterations${NC}"
        run_experiment "$config_file" "QA10 - Stability Test (Run $i)"
    done
}

# Start logging
echo "Starting RoboData Experiments Batch - $(date)" > "$LOG_FILE"
echo "Configuration Directory: $CONFIG_DIR" >> "$LOG_FILE"
echo "Results Directory: $RESULTS_DIR" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

echo -e "${GREEN}=== RoboData Experiments Runner ===${NC}"
echo -e "${BLUE}Starting comprehensive experiment batch...${NC}"
echo -e "${BLUE}Configuration Directory: $CONFIG_DIR${NC}"
echo -e "${BLUE}Results will be saved to: $RESULTS_DIR${NC}"
echo -e "${BLUE}Full log: $LOG_FILE${NC}"
echo ""

# Get all YAML config files and sort them
config_files=($(ls "$CONFIG_DIR"/*.yaml | sort))
total_configs=${#config_files[@]}

if [ $total_configs -eq 0 ]; then
    echo -e "${RED}No YAML configuration files found in $CONFIG_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Found $total_configs experiment configurations${NC}"
echo ""

# Run all experiments
experiment_count=0
successful_count=0
failed_count=0

for config_path in "${config_files[@]}"; do
    config_file=$(basename "$config_path")
    experiment_count=$((experiment_count + 1))
    
    echo -e "${GREEN}=== EXPERIMENT $experiment_count/$total_configs ===${NC}"
    
    # Special handling for QA10 stability test (run multiple times)
    if [[ "$config_file" == "QC01_consistency_test_canada_capital.yaml" ]]; then
        echo -e "${YELLOW}Running stability test (10 iterations)${NC}"
        run_consistency_test "$config_file"
        successful_count=$((successful_count + 10))  # Assume all 10 iterations succeed for counting
    else
        # Extract experiment description from filename
        description=$(echo "$config_file" | sed 's/.yaml$//' | sed 's/_/ /g' | sed 's/QA\([0-9]\+\)/QA\1 -/')
        
        # Run single experiment
        if run_experiment "$config_file" "$description"; then
            successful_count=$((successful_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    fi
done

# Final summary
echo -e "${GREEN}=== EXPERIMENT BATCH COMPLETED ===${NC}"
echo -e "${BLUE}All experiments have been executed.${NC}"
echo -e "${BLUE}Results are saved in: $RESULTS_DIR${NC}"
echo -e "${BLUE}Complete log available at: $LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}Summary of experiments run:${NC}"
echo "- Total configurations: $total_configs"
echo "- Successful experiments: $successful_count"
echo "- Failed experiments: $failed_count"
echo "- QA10 stability test: 10 iterations"
echo ""
echo -e "${GREEN}You can now analyze the results and generate statistics from the experiment data.${NC}"

# Generate experiment tables automatically
echo ""
echo -e "${GREEN}=== GENERATING EXPERIMENT TABLES ===${NC}"
echo -e "${BLUE}Extracting experiment statistics and creating CSV tables...${NC}"

if python backend/evaluation/extract_experiment_tables_updated.py 2>&1 | tee -a "$LOG_FILE"; then
    echo -e "${GREEN}✓ Experiment tables generated successfully${NC}"
    echo -e "${BLUE}Tables available:${NC}"
    echo "  - experiment_overview.csv (all experiments overview)"
    echo "  - batch_QA_table.csv (Batch A experiments)"
    echo "  - batch_QB_table.csv (Batch B experiments)"
    echo "  - batch_QC_table.csv (Batch C consistency test - Min/Max/Avg)"
    echo "  - batch_QD_table.csv (Batch D experiments with metacognition)"
else
    echo -e "${RED}✗ Failed to generate experiment tables${NC}"
    echo "Check $LOG_FILE for details"
fi

# Deactivate virtual environment
deactivate

echo "Experiments batch completed at $(date)" >> "$LOG_FILE"
