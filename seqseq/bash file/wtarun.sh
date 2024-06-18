python run_image_classification_no_trainer.py \
    --dataset_name "cats_vs_dogs" \
    --output_dir  /vit_foldercopy/wta_vit/seqseq/output/ \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --with_tracking \
    --do_train \
    --do_eval \
    --push_to_hub \
    --image_column_name "image" \
    --label_column_name "label"\
    --seed 1337

