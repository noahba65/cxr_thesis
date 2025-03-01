#!/bin/bash

# Loop through the model names (e.g., truncated_b0, truncated_b0_leaky)
for MODEL_NAME in "truncated_b0" "truncated_b0_leaky" "truncated_b0_leaky2"; do
    # Loop through truncated layers (0 to 5)
    for TRUNCATED_LAYERS in {0..4}; do
        # Loop over two cases: with and without --pretrained
        for PRETRAINED in true false; do

            echo "Running experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED"

            # Define the results directory based on whether pretrained is used
            if [ "$PRETRAINED" = "true" ]; then
                RESULTS_DIR="test/pretrained"
                CMD=("python" "run_model.py" "--pretrained")
            else
                RESULTS_DIR="test/not_pretrained"
                CMD=("python" "run_model.py")
            fi

            # Append other parameters
            CMD+=(
                "--model_name" "$MODEL_NAME"
                "--truncated_layers" "$TRUNCATED_LAYERS"  # Pass as string, but Python will parse it as int
                "--save_logs"
                "--epochs" "50"
                "--data_dir" "data_3_class"
                "--batch_size" "32"
                "--lr" ".001"
                "--results_folder_name" "$RESULTS_DIR"
                "--normalize"
                "--seed" "42"
            )

            # Execute the command
            "${CMD[@]}"

            echo "Experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED completed."
            echo "--------------------------------------------------"
        done
    done
done

echo "All experiments completed."