from fire import Fire
#from scripts.esrtGanSRnew import EsrtGanSR
from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR


def main(config="train"):
    app = EsrtGanSR(config)
    app.train()
    app = SrcnnSR(config)
    app = SrganSR(config)
    app = EfficientTransformerSR(config)
    app = EsrtGanSR(config)
    app.train()
    app = SrcnnSR(config)
    app.train()
    app = SrganSR(config)
    app.train()
    app = SrganSR(config)
    app.train()


if __name__ == '__main__':
    Fire(main)
