import numpy as np
import matplotlib.pyplot as plt


class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]


def display_train_stats(cfl_stats, communication_rounds):
    plt.figure(figsize=(4, 4))

    plt.subplot(1, 1, 1)
    acc_client_mean = np.mean(cfl_stats.acc_clients, axis=1)
    acc_client_std = np.std(cfl_stats.acc_clients, axis=1)
    plt.fill_between(cfl_stats.rounds, acc_client_mean - acc_client_std, acc_client_mean + acc_client_std, alpha=0.5,
                     color="C0")
    plt.plot(cfl_stats.rounds, acc_client_mean, color="C0")

    acc_server_mean = np.mean(cfl_stats.acc_servers, axis=1)
    acc_server_std = np.std(cfl_stats.acc_servers, axis=1)
    plt.fill_between(cfl_stats.rounds, acc_server_mean - acc_server_std, acc_server_mean + acc_server_std, alpha=0.5,
                     color="C1")
    plt.plot(cfl_stats.rounds, acc_server_mean, color="C1")

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")

    plt.xlim(0, communication_rounds)
    plt.ylim(0, 1)

    plt.show()
