pip install transformers

torchrun --nproc_per_node=2 main_pretrain.py \
--data_path /niddk-data-central/SOL/PASOS/train/SOL_10hz \
--batch_size 12 \
--world_size 2 \
--epochs 100 \
--warmup_epochs 10 \
--std_sampling \
--remark SOL \
--save_freq 2 \
--mask_ratio 0.5 \
--patch_emb 'sundial'

# chmod +x script/moca_pretrain_sol_new.sh
# ./script/moca_pretrain_sol_new.sh