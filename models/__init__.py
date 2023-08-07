def get_model(args, n_classes):
    if args.model == "convnet":
        from .ConvNet import ConvNet
        return ConvNet(in_channels=3, num_classes=n_classes)
    if args.model == "mnistnet":
        from .MnistNet import MnistNet
        return MnistNet(class_num=n_classes)
    if args.model == "regnet":
        from .RegNet import RegNetX_200MF
        return RegNetX_200MF(class_num=n_classes)
    if args.model == "resnet":
        from .Resnet import ResNet18
        return ResNet18(class_num=n_classes)
    if args.model == "mlp":
        from .MLP import MLP
        return MLP(784, 200, n_classes)
    if args.model == "vgg":
        from .VGG import VGG
        return VGG("VGG11star", class_num=n_classes)
    if args.model == "lenet":
        from .LeNet import LeNet5
        return LeNet5(num_classes=n_classes)
    raise ValueError("Unknown model")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
