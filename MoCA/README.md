# Meeting Record

## Apr 7th 

- Yu: The raw series is highly noisy, making direct correlation calculations unreliable. It’s better to perform these calculations on a higher-level representation. However, early network training often produces meaningless results. To address this, initializing MoCA can help establish a foundation for calculating covariance using meaningful representations.

- Yu: We should consider our masking method is a generic method that could apply on any task (like image)

- Yu: Complexity is a big issue.

TODO:

1. Init MoCA-20 epoch and then apply max-cut on high level representation
2. Ablatated the performance effect of max-iteration for local search algorithm

Future-TODO:
1. expand the pretrain dataset with NHANCE
2. use Capture-24 for downstream task

## Apr 14th
- latent level max-cut train from scratch is better than training from initing MoCA 20 epoch (eval through downstream task). This is counter intuitive.
- Use larger dataset

TODO:

1. try training with MoCA-100 epoch to see what happen
2. hyper parameter tuning on downstream set.
3. Preprocess Nhanes, maybe using 10Hz (like CHAP) to save storage space and also speed up training.
4. Preprocecss Capture-24 for evaluation

Future TODO:

1. Considering Mulitmodal (FLIP) pretraining to do ZeroShot
2. Considering constructing a large language-activity Dataset by generating series label by (1) and then apply some filtering algorithm
