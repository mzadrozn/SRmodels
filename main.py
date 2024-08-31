from fire import Fire
from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR


def main(config="train"):
    #prosty test działania
    app = SrcnnSR(config)
    app.train()


if __name__ == '__main__':
    Fire(main)
