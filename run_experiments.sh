#!/bin/bash

# Define the model name (e.g., b0, b1, b2, b3)
MODEL_NAME="b0"  # Change this to the desired model

# Loop through truncated layers (0 to 5)
for TRUNCATED_LAYERS in {0..5}; do
    # Loop over two cases: with and without --pretrained
    for PRETRAINED in true false; do
        
        echo "Running experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED"

        # Define the results directory based on whether pretrained is used
        if [ "$PRETRAINED" = true ]; then
            RESULTS_DIR="3_class_results/pretrained"
            CMD=("python" "run_model.py" "--pretrained")
        else
            RESULTS_DIR="3_class_results/not_pretrained"
            CMD=("python" "run_model.py")
        fi

        # Append other parameters
        CMD+=(
            "--model_name" "$MODEL_NAME"
            "--truncated_layers" "$TRUNCATED_LAYERS"
            "--save_logs"
            "--epochs" "40"
            "--data_dir" "data_3_class"
            "--batch_size" "32"
            "--lr" ".001"
            "--results_folder_name" "$RESULTS_DIR"
            "--bootstrap_n" "200"
            "--normalize"
            "--seed" "42"
        )

        # Execute the command
        "${CMD[@]}"

        echo "Experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED completed."
        echo "--------------------------------------------------"
    done
done

echo "All experiments completed."