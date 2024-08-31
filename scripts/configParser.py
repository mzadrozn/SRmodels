import yaml
from .utils import *
from pathlib import Path


class ConfigParser:
    def __init__(self, model, filename="train"):
        self.filename = f"{filename}.yaml" if filename[-5:] != ".yaml" else filename
        self.file = Path(PATHS.CONFIGS/model, self.filename)
        self.fileContent = self.readYaml(self.file) or dict()
        self.defaultContent = self.readYaml(PATHS.CONFIGS/model/"default.yaml") or dict()
        self.overrideContent = self.readYaml(PATHS.CONFIGS/model/"override.yaml") or dict()
        self.content = {**self.defaultContent, **self.overrideContent, **self.fileContent}

    def readYaml(self, path):
        if not path.exists():
            createFile(path)
            return dict()
        try:
            return yaml.safe_load(path.open("r+", encoding="utf-8"))
        except yaml.YAMLError as exc:
            print(f"<Warning> no {path.stem} file")
            return dict()


if __name__ == "__main__":
    a = ConfigParser("train.yaml")
