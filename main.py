import argparse
from train_and_test import batch
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deep Learning with Cifar-10 Datasets"
    )
    parser.add_argument("--model", required=True, type=str, help="choose model from mlp and cnn", default="mlp")
    parser.add_argument("--optimizer", required=True, type=str, help="choose optimizer from adam and sgd", default="sgd")
    parser.add_argument("--lr", required=True, type=str, help="specify learning rate")
    args = parser.parse_args()

    model = args.model
    optimizer=args.optimizer
    lr=args.lr

    batch(model = model, optimizer= optimizer, learning_rate= lr)
