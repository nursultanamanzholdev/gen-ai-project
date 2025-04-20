!torchrun --nproc_per_node=2 /kaggle/working/XTTSv2-Finetuning-for-New-Languages/train_gpt_xtts.py \
--use_ddp \
--output_path /kaggle/working/checkpoints/ \
--metadatas /kaggle/working/ISSAI_dataset_14Gb/datasets-F1/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-F1/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-F2/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-F2/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-F3/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-F3/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-M1_Book/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-M1_Book/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-M1_News/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-M1_News/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-M1_Wiki/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-M1_Wiki/metadata_eval.csv,kk \
           /kaggle/working/ISSAI_dataset_14Gb/datasets-M2/metadata_train.csv,/kaggle/working/ISSAI_dataset_14Gb/datasets-M2/metadata_eval.csv,kk \
--num_epochs 5 \
--batch_size 8 \
--grad_acumm 4 \
--max_text_length 500 \
--max_audio_length 441000 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 50000