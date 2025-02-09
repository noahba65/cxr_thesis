#!/bin/bash

# Define the model name (e.g., b0, b1, b2, b3)
MODEL_NAME="b0"  # Change this to the desired model

# Loop through truncated layers (0 to 5)
for TRUNCATED_LAYERS in {0..5}
do
    # Loop through pretrained options (true and false)
    for PRETRAINED in False
    do
        echo "Running experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED"

        # Run the Python script with the current parameters
        python run_model.py \
            --model_name "$MODEL_NAME" \
            --truncated_layers "$TRUNCATED_LAYERS" \
            --pretrained $PRETRAINED \
            --save_logs  \
            --epochs 40 \
            --data_dir "test_2_clxass" \
            --batch_size 32 \
            --lr .001 \
            --results_folder_name "2_class_results" \
            --bootstrap_n 200 \
            --data_dir "data_2_class" \
            --normalize True 
            
        



        echo "Experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED completed."
        echo "--------------------------------------------------"
    done
done

echo "All experiments completed."