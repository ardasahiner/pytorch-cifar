# Impact of Gated ReLU

Measuring impact of Gated ReLU v.s. ReLU in deep learning architectures on CIFAR-10. Currently only ResNet architecture is implemented.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py --gated_relu --data_dir = DATA_DIR

```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| ResNet18(Baseline)              | 95.49%    |
| ResNet18(Gated ReLU)          | 89.22%      |
