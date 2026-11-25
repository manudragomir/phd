# Vision Transformer readout experiments

CIFAR-10, 45,000 training images and 5,000 validation images taken from the official training set (seed 42). Every run below uses the same training loop: 100 epochs, batch size 256, AdamW starting at `3e-4`, 8 transformer blocks, image size 32. The learning rate stays fixed except on the row that names a step schedule. Unless a row says otherwise, the embedding size is 384 and there are 6 heads. The saved checkpoint is the weights after epoch 100. The best validation accuracy is the highest validation accuracy seen during training; those weights were not saved separately.

Up to four classifiers are trained from scratch for each setting. They differ only in the vector passed to the linear layer.

- **cls**: the class token
- **mean**: the average of the patch tokens
- **patch0**: the first patch token, the top-left patch
- **rand**: one patch token drawn again on every forward pass, from any patch of that image, never the class token. Only the two rows that name this readout were trained. Each reported accuracy is one draw, including the final validation report.

**Average** is the mean of the cls, mean, and patch0 numbers. It is not another trained model, and it does not include rand.

## Validation accuracy

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 77.12 | 78.12 | 77.96 |  | 77.73 |
| Patch 8, 1D positions, dropout 0.2 | 77.42 | 77.74 | 76.84 |  | 77.33 |
| Patch 8, 2D positions, dropout 0.1 | 76.54 | 78.46 | 78.02 |  | 77.67 |
| Patch 4, 1D positions, dropout 0.1 | 80.38 | 81.88 | 80.42 |  | 80.89 |
| Patch 4, 2D positions, dropout 0.1 | 80.06 | 81.74 | 81.54 |  | 81.11 |
| Patch 4, 2D positions, dropout 0.1, step LR | 81.10 | 82.50 | 81.48 | 81.14 | 81.69 |
| Patch 4, 2D positions, dropout 0.2 | 80.36 | 81.66 | 80.58 | 79.60 | 80.87 |
| Patch 4, 2D positions, dropout 0.3 | 79.88 | 80.62 | 80.44 |  | 80.31 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 77.38 | 77.32 | 77.60 |  | 77.43 |

Epoch 100, the saved model.

## Training accuracy at epoch 100

Same checkpoint as the validation table above. This is the accuracy on the training set, which keeps rising after the validation accuracy has leveled off.

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 96.35 | 95.38 | 94.92 |  | 95.55 |
| Patch 8, 1D positions, dropout 0.2 | 92.46 | 92.94 | 91.52 |  | 92.31 |
| Patch 8, 2D positions, dropout 0.1 | 96.15 | 95.65 | 94.75 |  | 95.52 |
| Patch 4, 1D positions, dropout 0.1 | 97.52 | 97.30 | 95.32 |  | 96.71 |
| Patch 4, 2D positions, dropout 0.1 | 97.43 | 97.53 | 96.18 |  | 97.05 |
| Patch 4, 2D positions, dropout 0.1, step LR | 91.23 | 92.91 | 89.43 | 89.47 | 91.19 |
| Patch 4, 2D positions, dropout 0.2 | 95.12 | 96.51 | 94.77 | 94.16 | 95.47 |
| Patch 4, 2D positions, dropout 0.3 | 91.77 | 95.36 | 91.09 |  | 92.74 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 93.80 | 92.29 | 91.55 |  | 92.55 |

## Best validation accuracy during training

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 78.00 (epoch 84) | 79.10 (epoch 76) | 78.52 (epoch 78) |  | 78.54 |
| Patch 8, 1D positions, dropout 0.2 | 78.00 (epoch 94) | 78.62 (epoch 92) | 78.52 (epoch 93) |  | 78.38 |
| Patch 8, 2D positions, dropout 0.1 | 77.32 (epoch 95) | 79.58 (epoch 72) | 78.86 (epoch 92) |  | 78.59 |
| Patch 4, 1D positions, dropout 0.1 | 81.12 (epoch 72) | 82.28 (epoch 84) | 81.98 (epoch 59) |  | 81.79 |
| Patch 4, 2D positions, dropout 0.1 | 81.00 (epoch 98) | 82.64 (epoch 63) | 82.30 (epoch 83) |  | 81.98 |
| Patch 4, 2D positions, dropout 0.1, step LR | 81.44 (epoch 69) | 82.92 (epoch 46) | 81.76 (epoch 60) | 81.50 (epoch 97) | 82.04 |
| Patch 4, 2D positions, dropout 0.2 | 81.28 (epoch 78) | 82.80 (epoch 87) | 82.06 (epoch 87) | 81.66 (epoch 72) | 82.05 |
| Patch 4, 2D positions, dropout 0.3 | 80.28 (epoch 67) | 82.28 (epoch 94) | 81.44 (epoch 81) |  | 81.33 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 77.70 (epoch 98) | 78.20 (epoch 70) | 77.96 (epoch 79) |  | 77.95 |

The average of the best epochs mixes three different epochs. It is a summary of each readout's peak, not the accuracy of one shared checkpoint.

Training accuracy at that same best-validation epoch:

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 94.91 | 92.03 | 91.60 |  | 92.85 |
| Patch 8, 1D positions, dropout 0.2 | 91.26 | 91.50 | 90.43 |  | 91.06 |
| Patch 8, 2D positions, dropout 0.1 | 95.80 | 92.00 | 93.96 |  | 93.92 |
| Patch 4, 1D positions, dropout 0.1 | 95.78 | 96.41 | 89.86 |  | 94.02 |
| Patch 4, 2D positions, dropout 0.1 | 97.35 | 94.70 | 94.51 |  | 95.52 |
| Patch 4, 2D positions, dropout 0.1, step LR | 90.80 | 90.22 | 88.87 | 89.24 | 89.96 |
| Patch 4, 2D positions, dropout 0.2 | 92.13 | 96.08 | 92.83 | 90.22 | 93.68 |
| Patch 4, 2D positions, dropout 0.3 | 85.24 | 95.08 | 87.54 |  | 89.29 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 93.65 | 86.99 | 87.62 |  | 89.42 |

## Validation macro F1 at epoch 100

Macro-averaged F1 from the validation classification report (`digits=4`) of the saved epoch-100 model. Each class contributes equally.

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 0.7682 | 0.7803 | 0.7794 |  | 0.7760 |
| Patch 8, 1D positions, dropout 0.2 | 0.7717 | 0.7766 | 0.7658 |  | 0.7714 |
| Patch 8, 2D positions, dropout 0.1 | 0.7632 | 0.7827 | 0.7789 |  | 0.7749 |
| Patch 4, 1D positions, dropout 0.1 | 0.8021 | 0.8186 | 0.8040 |  | 0.8082 |
| Patch 4, 2D positions, dropout 0.1 | 0.7995 | 0.8148 | 0.8153 |  | 0.8099 |
| Patch 4, 2D positions, dropout 0.1, step LR | 0.8097 | 0.8235 | 0.8131 | 0.8126 | 0.8154 |
| Patch 4, 2D positions, dropout 0.2 | 0.8025 | 0.8157 | 0.8040 | 0.7959 | 0.8074 |
| Patch 4, 2D positions, dropout 0.3 | 0.7976 | 0.8063 | 0.8031 |  | 0.8023 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 0.7719 | 0.7722 | 0.7747 |  | 0.7729 |

## Trainable parameters

The three readouts in one setting have the same count. The class token, the blocks, and the classifier do not change. Patch size and the position table do.

| Setting | Parameters |
|---|---:|
| Patch 8, 1D positions, dropout 0.1 or 0.2 | 14,272,138 |
| Patch 8, 2D positions, dropout 0.1 | 14,268,682 |
| Patch 4, 1D positions, dropout 0.1 | 14,235,274 |
| Patch 4, 2D positions, dropout 0.1, 0.2, or 0.3, including the step schedule and the random-patch readout | 14,216,458 |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 3,597,130 |

## Average time per epoch

Each number is the mean of the `Time:` field in that run's log. The last column is the mean of the three readout averages.

| Setting | cls | mean | patch0 | rand | average |
|---|---:|---:|---:|---:|---:|
| Patch 8, 1D positions, dropout 0.1 | 6.50 s | 6.42 s | 6.50 s |  | 6.47 s |
| Patch 8, 1D positions, dropout 0.2 | 7.09 s | 6.43 s | 6.43 s |  | 6.65 s |
| Patch 8, 2D positions, dropout 0.1 | 6.28 s | 6.39 s | 6.74 s |  | 6.47 s |
| Patch 4, 1D positions, dropout 0.1 | 14.52 s | 14.60 s | 14.51 s |  | 14.54 s |
| Patch 4, 2D positions, dropout 0.1 | 14.55 s | 14.59 s | 14.50 s |  | 14.55 s |
| Patch 4, 2D positions, dropout 0.1, step LR | 14.49 s | 14.58 s | 14.51 s | 14.44 s | 14.53 s |
| Patch 4, 2D positions, dropout 0.2 | 14.48 s | 14.55 s | 14.49 s | 14.41 s | 14.51 s |
| Patch 4, 2D positions, dropout 0.3 | 14.43 s | 14.51 s | 14.50 s |  | 14.48 s |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | 6.12 s | 6.25 s | 6.27 s |  | 6.21 s |

An epoch here includes the training pass and the validation pass.

## Where the files are

| Setting | Directories |
|---|---|
| Patch 8, 1D positions, dropout 0.1 | `runs/cls`, `runs/mean`, `runs/patch0` |
| Patch 8, 1D positions, dropout 0.2 | `runs/cls_dropout0.2`, `runs/mean_dropout0.2`, `runs/patch0_dropout0.2` |
| Patch 8, 2D positions, dropout 0.1 | `runs/cls_pos2d`, `runs/mean_pos2d`, `runs/patch0_pos2d` |
| Patch 4, 1D positions, dropout 0.1 | `runs/cls_patch4`, `runs/mean_patch4`, `runs/patch0_patch4` |
| Patch 4, 2D positions, dropout 0.1 | `runs/cls_patch4_pos2d`, `runs/mean_patch4_pos2d`, `runs/patch0_patch4_pos2d` |
| Patch 4, 2D positions, dropout 0.1, step LR | `runs/cls_patch4_pos2d_steplr25`, `runs/mean_patch4_pos2d_steplr25`, `runs/patch0_patch4_pos2d_steplr25`, `runs/rand_patch4_pos2d_steplr25` |
| Patch 4, 2D positions, dropout 0.2 | `runs/cls_patch4_pos2d_dropout0.2`, `runs/mean_patch4_pos2d_dropout0.2`, `runs/patch0_patch4_pos2d_dropout0.2`, `runs/rand_patch4_pos2d_dropout0.2` |
| Patch 4, 2D positions, dropout 0.3 | `runs/cls_patch4_pos2d_dropout0.3`, `runs/mean_patch4_pos2d_dropout0.3`, `runs/patch0_patch4_pos2d_dropout0.3` |
| Patch 8, 1D positions, embedding 192, dropout 0.1 | `runs/cls_emb192_patch8_pos1d`, `runs/mean_emb192_patch8_pos1d`, `runs/patch0_emb192_patch8_pos1d` |

Each directory has `log.txt`, `loss.png`, `accuracy.png`, and the checkpoint. Logs for the 1D dropout-0.1 runs do not print `pos_embed`; those models use the original flattened position table.

The step schedule multiplies `3e-4` by 0.1 after epochs 25, 50, and 75.

## Still to run

Nothing is training.
The earlier embedding-192 queue stopped at 15:44 on `runs/cls_emb192_patch4_pos1d/log.txt`, epoch 71 of 100 (validation accuracy 80.46%). Still unfinished if restarted later: finish that setting (`emb192_patch4_pos1d`), then `emb192_patch8_pos2d`, then `emb192_patch4_pos2d`. All dropout 0.1 and 6 heads. The embedding-size-128 runs were removed before they started.
