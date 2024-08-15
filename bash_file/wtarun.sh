python modeling/run_image_classification_no_trainer.py \
    --dataset_name cifar100 \
    --output_dir  ./output/cifar100_coarse/cifar100_ratio=0.3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 8 \
    --learning_rate 5e-5 \
    --with_tracking \
    --image_column_name img \
    --label_column_name coarse_label \
    --seed 1337
    

