export CUDA_VISIBLE_DEVICES=0
seeds=(2028 2027 2026 2023 2024 2025)

configs=(
    # "10 0.0 8"
    # "25 0.0 8"
    # "50 0.0 8"
    "100 1.0 15"
    # "200 1.0 15"
    # "400 1.0 15"
    # "600 1.0 15"
)

for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        read -r seq_len noise_std class_num <<< "$config"

        if [[ "$seq_len" -eq 10 || "$seq_len" -eq 25 ]]; then
            time_block_num=2
        elif [[ "$seq_len" -eq 50 || "$seq_len" -eq 100 ]]; then
            time_block_num=5
        elif [[ "$seq_len" -eq 200 ]]; then
            time_block_num=10
        elif [[ "$seq_len" -eq 400 ]]; then
            time_block_num=20
        elif [[ "$seq_len" -eq 600 ]]; then
            time_block_num=30
        else
            echo "Undefined seq_len: $seq_len, skipping..."
            continue
        fi

        echo "--------------------------------------------------------------"
        echo "Running job: DelayedCopy-Seed--${seed}--Seq_Len--${seq_len}--Noise_Std--${noise_std}--Class_Num--${class_num}--Slide_Num--${time_block_num}"
        echo "--------------------------------------------------------------"
        
        ./scripts/sequence_delay/Sptt_Window_Accum.sh "$seed" "$time_block_num" "$seq_len" "$noise_std" "$class_num"
        ./scripts/sequence_delay/TBptt_Accum.sh "$seed" "$time_block_num" "$seq_len" "$noise_std" "$class_num"

        ./scripts/sequence_delay/Sptt_Window_LastOnly.sh "$seed" "$time_block_num" "$seq_len" "$noise_std" "$class_num"
        ./scripts/sequence_delay/TBptt_LastOnly.sh "$seed" "$time_block_num" "$seq_len" "$noise_std" "$class_num"
    done
done