#!/bin/bash
MODEL="deit_small"
# Define the log file where all outputs will be stored
LOGFILE="experiment_log_${MODEL}.txt"

# Define the attn_quant schemes to iterate over
ATTN_QUANT_SCHEMES=( "Symmetric_UINT4" "Symmetric_UINT8" "Log2_half_Quantizer" "Log2Quantizer" "LogSqrt2Quantizer" "NoQuant" )

# Clear the log file if it already exists
> $LOGFILE

# Loop over each attn_quant scheme and run the Python script
for SCHEME in "${ATTN_QUANT_SCHEMES[@]}"
do
    echo "Running experiment with attn_quant=$SCHEME" | tee -a $LOGFILE
    python -u quant_train.py --attn_quant $SCHEME 2>&1 | tee -a $LOGFILE
    echo "-----------------------------------------------------" | tee -a $LOGFILE
done