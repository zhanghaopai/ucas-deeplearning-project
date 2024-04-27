import argparse
from train_and_test import batch
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deep Learning with Cifar-10 Datasets"
    )
    parser.add_argument("--model", required=True, type=str, help="choose model from mlp and cnn", default="mlp")
    parser.add_argument("--lr", required=True, type=str, help="specify learning rate")
    parser.add_argument("--optimizer", required=False, type=str, help="choose optimizer from adam and sgd", default="sgd")
    parser.add_argument("--active_function", required=False, type=str, help="specify active function", default='relu')
    args = parser.parse_args()

    model = args.model
    optimizer=args.optimizer
    lr=args.lr
    af=args.af

    batch(model = model, optimizer= optimizer, learning_rate= lr, active_function= af)
