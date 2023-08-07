# 3SFC
The official code for "Communication-efficient Federated Learning with Single-Step Synthetic Features Compressor for Faster Convergence", ICCV 2023

## Abstract
> Reducing communication overhead in federated learning (FL) is challenging but crucial for large-scale distributed privacy-preserving machine learning. While methods utilizing sparsification or other techniques can largely reduce the communication overhead, the convergence rate is also greatly compromised. In this paper, we propose a novel method named Single-Step Synthetic Features Compressor (3SFC) to achieve communication-efficient FL by directly constructing a tiny synthetic dataset containing synthetic features based on raw gradients. Therefore, 3SFC can achieve an extremely low compression rate when the constructed synthetic dataset contains only one data sample. Additionally, the compressing phase of 3SFC utilizes a similarity-based objective function so that it can be optimized with just one step, considerably improving its performance and robustness. To minimize the compressing error, error feedback (EF) is also incorporated into 3SFC. Experiments on multiple datasets and models suggest that 3SFC has significantly better convergence rates compared to competing methods with lower compression rates (i.e., up to 0.02\%). Furthermore, ablation studies and visualizations show that 3SFC can carry more information than competing methods for every communication round, further validating its effectiveness.

## Examples

Train MLP with MNIST using 3SFC on a cluster containing 10 clients.

```bash
python main.py --method ours --n_client 10 --n_epoch 200 --n_client_epoch 5 --dataset mnist --batch_size 64 --lr 1e-2 --model mlp --ours_n_sample 1
```

## Citation
@inproceedings{zhou20233sfc,
  title={Communication-efficient Federated Learning with Single-Step Synthetic Features Compressor for Faster Convergence},
  author={Yuhao Zhou, Mingjia Shi, Yuanxi Li, Yanan Sun, Qing Ye, Jiancheng Lv},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
