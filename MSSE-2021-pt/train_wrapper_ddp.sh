rm -r /niddk-data-central/chap_animesh/training_results/results_nov26/checkpoints/train_ddp
cd /niddk-data-central/chap_animesh/DeepPostures/MSSE-2021-pt
# Preprocess
python train_model_ddp.py \
--pre-processed-dir /niddk-data-central/P2/pre_processed_pt \
--split-data-file /niddk-data-central/P2/CHAP2.0_support_files/P2_train_test_rand.csv \
--transfer-learning-model CHAP_ALL_ADULTS \
--num-epochs 2 \
--batch-size 32 \
--training-data-fraction 1 \
--validation-data-fraction 1 \
--model-checkpoint-path /niddk-data-central/chap_animesh/training_results/results_nov26/checkpoints/train_ddp \
--output-file /niddk-data-central/chap_animesh/training_results/results_nov26/outputs/train_ddp.csv | tee -a /niddk-data-central/chap_animesh/training_results/results_nov26/logs/train_ddp.log
