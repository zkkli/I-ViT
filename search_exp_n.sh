#!/bin/bash

# Define the log file where all outputs will be stored
LOGFILE="search_logs_vit_base.txt"

# Define the attn_quant schemes to iterate over
ATTN_QUANT_SCHEMES=("NoQuant")

# Define the exp_n values for intsoftmax_exp_n and intgelu_exp_n
SOFTMAX_EXP_VALUES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
GELU_EXP_VALUES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

# Clear the log file if it already exists
> $LOGFILE

# Loop over each attn_quant scheme, intsoftmax_exp_n, and intgelu_exp_n values
for SCHEME in "${ATTN_QUANT_SCHEMES[@]}"
do
    for SOFTMAX_EXP in "${SOFTMAX_EXP_VALUES[@]}"
    do
        for GELU_EXP in "${GELU_EXP_VALUES[@]}"
        do
            echo "Running experiment with attn_quant=$SCHEME, intsoftmax_exp_n=$SOFTMAX_EXP, intgelu_exp_n=$GELU_EXP" | tee -a $LOGFILE
            python quant_train.py --attn_quant $SCHEME --intsoftmax_exp_n $SOFTMAX_EXP --intgelu_exp_n $GELU_EXP 2>&1 | tee -a $LOGFILE
            echo "-----------------------------------------------------" | tee -a $LOGFILE
        done
    done
done

