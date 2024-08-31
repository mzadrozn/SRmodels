from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR
import argparse


def train_model(model_name, config="train"):
    model = None
    if model_name == 'ESRTGAN':
        model = EsrtGanSR(config)
    elif model_name == 'SRCNN':
        model = SrcnnSR(config)
    elif model_name == 'SRGAN':
        model = SrganSR(config)
    elif model_name == 'ESRT':
        model = EfficientTransformerSR(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Nazwa modelu", default="SRCNN")
    args = parser.parse_args()
    train_model(args.model_name)


if __name__ == "__main__":
    main()