MODEL_NAME = "ESRTGAN"


class AppSR:
    def __init__(self):
        self.model = None
        self.device = None
        self.modelG = None

    def train(self, loader=None):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def getModelName(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def loadBest(self):
        raise NotImplementedError()

