import argparse
import random


def get_config():
    parser = argparse.ArgumentParser(
        description="Federated Averaging Experiments")
    parser.add_argument("--method", type=str, default="ours")
    parser.add_argument("--n_client", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=1.0)
    parser.add_argument("--dirichlet", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--n_client_epoch", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="convnet")
    parser.add_argument("--seed", type=int, default=random.randint(1, 10000))

    parser.add_argument("--ours_n_sample", type=int, default=1)
    parser.add_argument("--topk", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    return args
