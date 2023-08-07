import time
from copy import deepcopy
import random
import torch
from torch.utils.data import DataLoader

synthetic_bs = 1
recursive_steps = 1


def train_op(model, loader, optimizer, epochs=1, loss_fn=torch.nn.CrossEntropyLoss(), device="cpu"):
    model.train()
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = loss_fn(model(x), y)

            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()

    return running_loss / samples


def eval_op(model, loader, device="cpu"):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct / samples


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def add_(target, added, addend):
    for name in target:
        target[name].data = added[name].data.clone() + addend[name].data.clone()


def weighted_add_(target, added, addend, weight):
    for name in target:
        target[name].data = weight * added[name].data.clone() + (1.0 - weight) * addend[name].data.clone()


def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp


def reduce_add_sum(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.sum(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data, device="cpu"):
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.device = device

    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader, device=self.device)


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, args, batch_size=128, train_frac=0.8, device="cpu"):
        super().__init__(model_fn, data, device)
        self.optimizer = optimizer_fn(self.model.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.data = data
        self.args = args
        self.n_train = int(len(data) * train_frac)
        self.n_eval = len(data) - self.n_train
        data_train, data_eval = torch.utils.data.random_split(self.data, [self.n_train, self.n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)

        self.id = idnum

        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_residual = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)

    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"] *= 0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs=epochs, loss_fn=self.loss_fn, device=self.device)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        add_(target=self.dW_residual, added=self.dW_residual, addend=self.dW)
        return train_stats

    def compute_fedsynth(self, n_sample, n_classes, eta_w, eta, epochs=1, loader=None):
        synthetic_input_size = [20, n_sample] + list(next(iter(self.train_loader))[0].shape[1:])
        synthetic_inputs = torch.randn(tuple(synthetic_input_size), device=self.device, requires_grad=True)
        synthetic_labels = torch.randn((20, n_sample, n_classes), device=self.device, requires_grad=True)

        optimizer = torch.optim.SGD([synthetic_inputs, synthetic_labels], lr=eta, momentum=0.0)

        for ep in range(epochs):
            synthetic_model = deepcopy(self.model)
            synthetic_model.train()
            synthetic_optim = torch.optim.SGD(synthetic_model.parameters(), lr=eta_w, momentum=0.0)
            for i in range(20):
                synthetic_optim.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(synthetic_model(synthetic_inputs[i]), synthetic_labels[i])
                loss.backward()
                print("step1", loss.item())
                synthetic_optim.step()

            for x, y in self.train_loader if not loader else loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                loss = torch.nn.CrossEntropyLoss()(synthetic_model(x), y)
                print("step2", loss.item())
                loss.backward()
                optimizer.step()

        # replay
        synthetic_model = deepcopy(self.model)
        synthetic_model.train()
        synthetic_optim = torch.optim.SGD(synthetic_model.parameters(), lr=eta_w, momentum=0.0)
        for i in range(20):
            synthetic_optim.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(synthetic_model(synthetic_inputs[i]), synthetic_labels[i])
            loss.backward()
            synthetic_optim.step()
        synthetic_W = {key: value for key, value in synthetic_model.named_parameters()}

        synthetic_gradients_flatten = torch.cat([self.W[k].clone().flatten() - v.clone().flatten() for k, v in synthetic_W.items()])
        real_gradients = torch.cat([v.flatten() for v in deepcopy(self.dW).values()])
        cos = torch.sum(synthetic_gradients_flatten * real_gradients) / (
                torch.norm(synthetic_gradients_flatten) * torch.norm(real_gradients) + 1e-12)

        scale_factor = cos * torch.norm(real_gradients) / torch.norm(synthetic_gradients_flatten)
        if torch.isnan(scale_factor):
            scale_factor = 0.0
        else:
            scale_factor = scale_factor.item()

        print(
            f"Client {self.id}: cos: {cos}, real norm: {torch.norm(real_gradients):.4f}, scale_factor: {scale_factor:.4f}")
        return deepcopy(synthetic_inputs), deepcopy(synthetic_labels), scale_factor, cos

    def compute_synthetic_sample(self, n_sample, n_classes):
        synthetic_input_size = [n_sample] + list(next(iter(self.train_loader))[0].shape[1:])
        synthetic_inputs = torch.randn(tuple(synthetic_input_size), device=self.device, requires_grad=True)
        synthetic_labels = torch.randn((n_sample, n_classes), device=self.device, requires_grad=True)

        synthetic_model = deepcopy(self.model)
        synthetic_model.eval()

        optimizer = torch.optim.LBFGS([synthetic_inputs, synthetic_labels])

        best_inputs, best_labels, best_loss = synthetic_inputs.clone(), synthetic_labels.clone(), float("inf")
        # s2 = torch.cat([v.clone().flatten() for v in self.dW_residual.values()])
        s2 = torch.cat([v.clone().flatten() for v in self.dW.values()])
        for iters in range(10):
            def closure():
                optimizer.zero_grad()
                synthetic_preds = synthetic_model(synthetic_inputs)
                loss = torch.nn.CrossEntropyLoss()(synthetic_preds, synthetic_labels)
                dy_dx = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True, allow_unused=True)

                s1 = torch.cat([v.flatten() for v in dy_dx])
                # grad_loss = 1. - torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)
                grad_loss = 1.0 - torch.abs(torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12))

                # grad_loss += 1e-5 * (torch.mean(synthetic_inputs ** 2) + torch.mean(synthetic_labels ** 2))
                grad_loss.backward()
                return grad_loss

            optimizer.step(closure)
            current_loss = closure()
            if 0 <= current_loss.item() < best_loss:
                best_inputs = synthetic_inputs.clone()
                best_labels = synthetic_labels.clone()
                best_loss = current_loss.item()

        # replay
        preds = synthetic_model(best_inputs)
        loss = torch.nn.CrossEntropyLoss()(preds, best_labels)
        synthetic_gradients = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True)

        synthetic_gradients_flatten = torch.cat([v.clone().flatten() for v in synthetic_gradients])
        # real_gradients = torch.cat([v.flatten() for v in deepcopy(self.dW_residual).values()])
        real_gradients = torch.cat([v.flatten() for v in deepcopy(self.dW).values()])
        cos = torch.sum(synthetic_gradients_flatten * real_gradients) / (
                    torch.norm(synthetic_gradients_flatten) * torch.norm(real_gradients) + 1e-12)

        scale_factor = cos * torch.norm(real_gradients) / torch.norm(synthetic_gradients_flatten)
        if torch.isnan(scale_factor):
            scale_factor = 0.0
        else:
            scale_factor = scale_factor.item()

        # synthetic_gradients_dict = {name: synthetic_gradients[i] * scale_factor for i, name in
        #                             enumerate(self.dW_residual)}
        #
        # subtract_(target=self.dW_residual, minuend=self.dW_residual, subtrahend=synthetic_gradients_dict)
        print(
            f"Client {self.id}: cos: {cos}, real norm: {torch.norm(real_gradients):.4f}, scale_factor: {scale_factor:.4f}")

        return best_inputs, best_labels, scale_factor, cos

    def compute_topk(self, p):
        real_gradients_flatten = torch.cat([v.flatten() for v in deepcopy(self.dW_residual).values()])

        # evaluate topk gradients
        topk_dW = {}
        for key, value in deepcopy(self.dW_residual).items():
            n_pick = max(int(value.numel() * p), 1)  # will pass both gradients and indexes.
            _, topk_i = torch.topk(torch.abs(value.flatten()), n_pick)
            topk = torch.zeros_like(value.flatten())
            topk[topk_i] = value.flatten()[topk_i]
            topk_dW[key] = topk.reshape(value.shape)
        topk_gradients_flatten = torch.cat([v.flatten() for v in deepcopy(topk_dW).values()])
        topk_cos = torch.sum(topk_gradients_flatten * real_gradients_flatten) / (
                torch.norm(topk_gradients_flatten) * torch.norm(real_gradients_flatten) + 1e-12)

        subtract_(target=self.dW_residual, minuend=self.dW_residual, subtrahend=topk_dW)

        return topk_dW, topk_cos

    def reset(self):
        copy(target=self.W, source=self.W_old)


class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, args, device="cpu"):
        super().__init__(model_fn, data, device)
        self.eval_loader = DataLoader(data, batch_size=128, shuffle=False)
        self.model_cache = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients) * frac))

    def aggregate(self, clients):
        reduce_add_average(targets=[self.W], sources=[client.dW for client in clients])

    def aggregate_fedsynth(self, synthetics):
        pass

    def aggregate_synthetic_gradients(self, synthetics, scale_factors, dws):
        client_gradients = []
        synthetic_model = deepcopy(self.model)

        for i, (inputs, labels) in enumerate(synthetics):
            preds = synthetic_model(inputs)
            loss = torch.nn.CrossEntropyLoss()(preds, labels)
            gradients = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True)
            # gradients_flatten = torch.cat([v.flatten() for v in deepcopy(gradients)])
            gradients = [scale_factors[i] * g.clone().detach() for g in gradients]

            gradients_dict = {name: gradients[i] for i, name in enumerate(self.W)}

            client_gradients.append(gradients_dict)

        tmp1 = {name: torch.mean(torch.stack([g[name] for g in client_gradients]), dim=0) for name in self.W}
        tmp2 = {name: torch.mean(torch.stack([g[name] for g in dws]), dim=0) for name in self.W}
        s1 = torch.cat([v.flatten() for v in deepcopy(tmp1).values()])
        s2 = torch.cat([v.flatten() for v in deepcopy(tmp2).values()])
        sim = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

        print("God View: ", sim, torch.norm(s1), torch.norm(s2), torch.norm(s1 - s2), (s1 - s2).abs().max())

        reduce_add_average(targets=[self.W], sources=client_gradients)

    def aggregate_fusion(self, dws):
        reduce_add_average(targets=[self.W], sources=dws)

    def aggregate_sign_compression(self, clients):
        sign_dWs = []
        scale_factors  = []
        for client in clients:
            scale_factor = torch.norm(torch.cat([v.flatten() for v in deepcopy(client.dW).values()])) / torch.norm(
                torch.cat([torch.sign(v.flatten()) for v in deepcopy(client.dW_residual).values()]))
            scale_factors.append(scale_factor.clone().detach())
            sign_dWs.append({key: torch.sign(value) * scale_factor for key, value in client.dW_residual.items()})
            subtract_(target=client.dW_residual, minuend=client.dW_residual, subtrahend=sign_dWs[-1])
        print("scale factors: ", scale_factors)
        reduce_add_average(targets=[self.W], sources=sign_dWs)

    def aggregate_topk_compression(self, clients, p=0.01):
        topk_dWs = []
        for client in clients:
            topk_dW = {}
            for key, value in client.dW_residual.items():
                n_pick = max(int(value.numel() * p), 1)    # will pass both gradients and indexes.
                _, topk_i = torch.topk(torch.abs(value.flatten()), n_pick)
                topk = torch.zeros_like(value.flatten())
                topk[topk_i] = value.flatten()[topk_i]
                topk_dW[key] = topk.reshape(value.shape)

            topk_dWs.append(topk_dW)
            subtract_(target=client.dW_residual, minuend=client.dW_residual, subtrahend=topk_dWs[-1])

        reduce_add_average(targets=[self.W], sources=topk_dWs)

    def aggregate_stc_compression(self, clients, p=0.01):
        topk_dWs = []

        total_size = 0
        compressed_size = 0
        for client in clients:
            topk_dW = {}

            for key, value in client.dW_residual.items():
                n_pick = max(int(value.numel() * p - (value.numel() * (1 - p)) / 31), 1)    # will pass both gradients and indexes.
                total_size += value.numel()
                compressed_size += n_pick + (value.numel() - n_pick) / 32

                topk_v, topk_i = torch.topk(torch.abs(value.flatten()), n_pick)
                n_topk_i = torch.abs(value.flatten()) < topk_v[-1]
                n_topk_mean = torch.mean(value.flatten()[n_topk_i])
                print("value", key, value.numel())

                topk = torch.zeros_like(value.flatten())
                topk[topk_i] = value.flatten()[topk_i]
                topk[n_topk_i] = torch.abs(n_topk_mean) * torch.sign(value.flatten()[n_topk_i])

                topk_dW[key] = topk.reshape(value.shape)

            topk_dWs.append(topk_dW)
            subtract_(target=client.dW_residual, minuend=client.dW_residual, subtrahend=topk_dWs[-1])

        print("compression ratio: ", compressed_size / total_size)

        reduce_add_average(targets=[self.W], sources=topk_dWs)