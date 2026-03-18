# Transformer Tuning Assignment Report
**Roll No:** B23BB1032

## 1. Baseline Metrics
- **Final Epochs:** 100
- **Final Loss:** 0.1019
- **BLEU Score:** 87.99
- **Training Time:** ~136 minutes

## 2. Hyperparameters Defined for Tuning
The following hyperparameters were explored using `OptunaSearch` and `ASHAScheduler`:
- **Learning Rate (`lr`):** `tune.loguniform(1e-5, 1e-3)`
- **Batch Size (`batch_size`):** `tune.choice([16, 32, 64])`
- **Number of Attention Heads (`num_heads`):** `tune.choice([4, 8])`
- **Feed-Forward Dimension (`d_ff`):** `tune.choice([1024, 2048])`
- **Dropout Rate (`dropout`):** `tune.uniform(0.1, 0.4)`

## 3. Best Configuration
The sweep successfully identified the following optimal configuration:
- **Best Learning Rate**: 0.0001
- **Best Batch Size**: 32
- **Best Number of Heads**: 4
- **Best d_ff**: 2048
- **Best Dropout**: 0.1013
- **Number of Layers**: 6

## 4. Final Metrics
- **Time taken for tuning:** ~2.5 hours (20 trials)
- **Final Loss of Best Model:** 0.1018 (at epoch 49)
- **Final BLEU Score:** 42.16 (Evaluated on sample set)
- **Epochs to beat baseline:** 49
- **Weight file saved as:** `B23BB1032_ass_4_best_model.pth`
