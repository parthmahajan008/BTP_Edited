python finetune_student.py \
    --data_path path/to/augmented_data.json \
    --model_name facebook/opt-350m \
    --output_dir path/to/save/model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4