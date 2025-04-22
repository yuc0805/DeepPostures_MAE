# Experiment Results - 02-27-2025

## Correlation Mask

The correlation mask uses the covariance matrix between patches (in raw series) to obtain a mask. Specifically, it sums all the pairwise covariance to determine which patch is most important, then masks accordingly.

- **MoCA 20 epoch**: `/home/jovyan/persistent-data/MAE_Accelerometer/experiments/2971(p200_10_alt_0.0005_both)/checkpoint-19.pth`
- **MoCA 200 epoch**: `/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth`
- **MoCA w/o Aug** `/home/jovyan/persistent-data/MAE_Accelerometer/experiments/661169(p200_10_alt_0.0005)/checkpoint-3999.pth`

- **MAE Synchronized Masking**:  `/home/jovyan/persistent-data/leo/output_dir/syncmask_8_checkpoint-200.pth`
- `/home/jovyan/persistent-data/MAE_Accelerometer/experiments/28533(p200_10_syn_0.0005_both)/checkpoint-19.pth`
- `/home/jovyan/persistent-data/MAE_Accelerometer/experiments/185(p200_10_syn_0.0005)/checkpoint-3999.pth`
----
- **Unmask important patch**: `covariance_raw_reverse_mask_2025-02-26_08-39`(200pth)
- **Mask important patch**: `covariance_raw_mask_2025-02-21_19-03`(200pth)
- **Spectral Mask**: `/home/jovyan/persistent-data/leo/spectral_mask_checkpoint-50.pth`

- **half-half mask**: `/home/jovyan/persistent-data/leo/optim_mask/ckpt/covariance_raw_split_mask_2025-03-16_16-16/covariance_raw_split_mask_checkpoint-200.pth`

-**Feat_maxcut** : 
    - 100iter: `/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_MoCA_20_2025-04-08_01-47/feat_maxcut_100iter_init_MoCA_20_checkpoint-200.pth`
    - 500iter: 
    - 1000iter:  `/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_1000iter_init_MoCA_20_2025-04-08_01-01/feat_maxcut_1000iter_init_MoCA_20_checkpoint-50.pth` (WAITING FOR MORE EPOCH)
                
-**Raw_maxcut**: 
    - 100iter: 
    - 500iter: `/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_500iter_2025-04-04_00-51/maxcut_500iter_checkpoint-200.pth`
    - 1000iter: `/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-200.pth`

## Generative Task

| Model/Task                                  | Random Masking (MAE/MSE) | Temporal Imputation (MAE/MSE) | Sensor Imputation (MAE/MSE) |
|---------------------------------------------|--------------------------|-------------------------------|-----------------------------|
| `MAE_mae`                                   | 0.0741 / 0.0284          | 0.1559 / 0.1079               | 0.123 / 0.0561              |
| `correlation_mask (unmask important patch)` | 0.0908 / 0.0414          | 0.1585 / 0.1107               | 0.156 / 0.0796              |
| `correlation_mask (mask important patch)`   | 0.1350 / 0.0645          | 0.2555 / 0.1863               | 0.1929 / 0.10183            |
| `Spectral Mask`                             | 0.168 / 0.080            | 0.2160 / 0.1430               | 0.175 / 0.087               |
| `Max-cut Mask`                              | 0.081 / 0.033            | 0.1537 / 0.1098               | 0.147 / 0.079               |
| `correlation_mask (half-half)`              | 0.081 / 0.033            | 0.1635 / 0.1136               | 0.1415 / 0.071              |


| **Model / Error (MAE / MSE) ↓**                        | **Random Imputation** | **Temporal Imputation** | **Sensor Imputation** | **Temporal Extrapolation** |
|--------------------------------------------------------|------------------------|--------------------------|------------------------|-----------------------------|
| Linear Interp.                                         | 0.587 / 0.689          | 0.546 / 0.634            | 0.252 / 0.189          | 0.415 / 0.474               |
| Nearest Neighbor                                       | 0.341 / 0.284          | 0.240 / 0.180            | 0.412 / 0.377          | 0.210 / 0.158               |
| MICE                                                   | 0.289 / 0.251          | 0.410 / 0.375            | 0.210 / 0.158          | 0.210 / 0.158               |
| MAE Synchronized Masking (w/o aug.)                    | 0.275 / 0.162          | 0.262 / 0.191            | 0.288 / 0.170          | 0.262 / 0.191               |
| MAE Synchronized Masking (w/ aug.)                     | 0.380 / 0.281          | 0.306 / 0.231            | 0.494 / 0.459          | 0.175 / 0.118               |
| **MoCA (w/o aug., Ours)**                              | 0.086 / 0.043     | **0.112 / 0.059**        | 0.160 / 0.089      | **0.112 / 0.060**           |
| **MoCA (w/ aug., Ours)**                               | **0.076 / 0.027**      | 0.153 / 0.102            | **0.140 / 0.063**      | 0.157 / 0.109               |


|Model(Epoch)|UCIHAR|WISDM|IMWSHA|Oppo|
|------------|------|-----|------|------|
|   vit w/ imagenet   | -      | - | - |71.73 /61.21      |
|Sync(No Augmentation)|86.30/91.70|86.60/93.31|57.14/57.14|71.73/73.83|
|Sync(20e)|88.40/91.30|90.73/92.15|60.30/61.90|73.36/73.83|
|SyncMask(200e)|92.31/94.80|88.80/93.60|58.70/60.32|74.07/79.91|
|Harnet10|87.60/93.10|88.40/96.50|57.10/71.43|75.93/83.41|
|SimSiam-Res50|53.10/88.50|79.90/34.90|47.60/44.41|-/-|
|BEiT(w/ visual codebook pretrained on Imagenet-1k)|86.10/70.40|86.90/75.60|60.30/30.21|71.73/60.05|
|BEiT(w/ codebook pretrained on UCI-HAR)|87.01/83.20|86.90/73.80|60.30/30.23|71.73/63.55|
|MoCA w/ random init|80.10/87.50|90.10/93.60|55.62/55.62|70.79/71.96|
|MoCA(No Augmentation)|92.40/94.80|90.70/91.57|52.40/57.10|71.73/77.80|
|MoCA(20e)|93.10/96.60|85.20/94.19|63.50/65.08|76.64/79.91|
|MoCA(200e)|91.40/96.90|93.60/96.22|76.20/71.43|77.34/81.54|

| Model/Dataset (LP/FT)                       | UCIHAR (7 classes) |
|---------------------------------------------|--------------------|
| `BEiT (w/ visual codebook pretrained on Imagenet-1k)`             | 86.10 / 70.4       |
| `BEiT (w/ time series codebook pretrained on UCI-HAR) - 80 epoch` | 87.01 / 83.23      |
| `BEiT (w/ time series codebook pretrained on UCI-HAR) - 640 epoch`| 82.60 / 83.23      |

### Correlation Mask Classification Task

### Need Rerun Results

| Model/Dataset (LP/FT)                       | UCIHAR (7 classes) | WISDM         | IMWSHA        | Oppoturnity   | Capture24 (4 class) 
|---------------------------------------------|--------------------|---------------|---------------|---------------|---------------|
|MoCA w/ random init                          |80.10/87.50|90.10/93.60|55.62/55.62|70.79/71.96| - / - |
|Harnet10                                     |87.60/93.10         |88.40/96.50    |57.10/71.43    | 75.93 / 83.41 | - / - |
| `MoCA (200e)`                               | 91.40 / 96.90      | 93.60 / 96.22 | 76.20 / 71.43 | 77.34 / 81.54 | 76.43 / - |
| `Raw Maxcut  (1000iter,200pth,scratch)`     | 94.83 / 95.84      | 95.06 / 95.35 | 69.84 / 76.19 | 80.14 / 80.37 | - / - |
| `Raw Maxcut  (500iter,200pth,scratch)`      | 94.83/ 95.71       | 94.48 / 95.06 | 73.02 / 76.19 | 77.57 / 80.61 | - / - |
| `Raw Maxcut  (100iter,200pth,scratch)`      |
|`feat_maxcut  (100iter, 200pth, MoCA20) `    | 93.69 / 95.08      | 93.31 / 95.06 | 73.02 / 69.84 | 80.37 / 79.91 | - / - |
|`feat-maxcut (500iter, 200pth, MoCA20)`      |
|`feat-maxcut (1000iter, 200pth, MoCA20)`     |
|`feat-maxcut (100iter, 200pth, scratch)`     | 93.82 /96.09       | 95.35/95.06   |76.19/68.25    |76.87/83.18    | - / - |
|`feat-maxcut (500iter, 200pth, scratch)`     |
| `Correlation Mask (Half-Half)`              | 94.20 / 95.71      | 93.90 / 94.77 | 73.02 / 80.95 |76.40 / 80.61  | - / - |


<!-- |   Below need rerun                    | -      | - | - |-     |-->
<!-- | `Correlation Mask (Unmask Important Patch)` | 93.82 / 94.45      | 91.86 / 93.60 | 68.25 / 66.67 | - / -         | -->
<!-- | `Correlation Mask (Mask Important Patch)`   | 79.82 / 73.39      | - / -         | - / -         | - / -         | -->
<!-- | `Spectral Mask (50pth)`                     | 66.71 / 62.67      | - / -         | - / -         | - / -         | -->
<!-- | `Maxcut Mask (100pth)`                      | 93.82 / 96.47      | 93.60 / 93.02 | 71.43 / 69.84 | - / -         | -->

<!-- | `Maxcut Mask (300pth)`                      | 93.57 / 96.60      | 95.93 / 96.22 | 68.25 / 80.95 | 79.91 / 81.07 | -->
Some observation:
For Maxcut, linear probing is ourperforming other model very well, but in finetune it's not that promising. But in general, linear probing is a clearer indication of good representation. Finetuning is prone to hyper-parameter tuning and overfitting. The WISDM and IMWSHA dataset is too small, need a larger dataset for finetune.

- analyze why imwsha lp poor for maxcut mask 200epoch


Some further analysis idea:
- choose the best model, and run how pretrained epochs effect downstream performance (some sort of scaling law analysis)
- Maxcut could be have a much bigger contribution, many mlm task could use it not only time series. If resource permit, we defintely should try it on ImageNet, in that way it would be a potential oral-paper. 

