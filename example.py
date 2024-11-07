import numpy as np

from typing import Any
from malwareDetector.detector import detector
from malwareDetector.config import write_config_to_file, read_config

class subDetector(detector):
    def __init__(self) -> None:
        super().__init__()
        #self.config = read_config(config_file_path="config2.json")

        # config: Config = Config()
        self.config.set_param("test123", 123)
        print(f"test123 = {self.config.test123}")
        self.config.del_param("test123")
        # classify: bool = DEFAULT_CLASSIFY
        print(f"classify = {self.config.classify}")
        # path: PathConfig = PathConfig()
        print(f"input path = {self.config.path.input}")
        print(f"output path = {self.config.path.output}")
        print(f"config path = {self.config.path.config}")
        self.config.path.set_param("test", "TEST_PATH")
        print(f"test path = {self.config.path.test}")
        self.config.path.del_param("test")
        # folder: FolderConfig = FolderConfig()
        print(f"dataset folder = {self.config.folder.dataset}")
        print(f"feature folder = {self.config.folder.feature}")
        print(f"vectorize folder = {self.config.folder.vectorize}")
        print(f"model folder = {self.config.folder.model}")
        print(f"predict folder = {self.config.folder.predict}")
        print(f"folder list = {self.config.folder.folder_list}")
        print(f"Create a new folder, the variable name is test, and the folder name is TEST_DIR")
        self.config.folder.set_folder("test", "TEST_DIR")
        self.mkdir()
        print(f"test folder = {self.config.folder.test}")
        print(f"folder list = {self.config.folder.folder_list}")
        print(f"Detele the TEST_DIR folder")
        self.config.folder.del_folder("test")
        print(f"folder list = {self.config.folder.folder_list}")
        # model: ModelConfig = ModelConfig()
        print(f"Create a new model variable, the variable name is gamma, and the value is 0.")
        self.config.model.set_param("gamma", 0)
        print(f"model gamma = {self.config.model.gamma}")
        print(f"Detele the gamma variable.  ")
        self.config.model.del_param("gamma")
        try:
            print(f"model gamma = {self.config.model.gamma}")
        except AttributeError as e:
            print(f"AttributeError: {e}")
        # wirte the config to file
        write_config_to_file(self.config, "config2.json")

    def extractFeature(self) -> Any:
        return 'This is the implementation of the extractFeature function from the derived class.'

    def vectorize(self) -> np.array:
        return 'This is the implementation of the vectorize function from the derived class.'

    def model(self) -> Any:
        return 'This is the implementation of the model function from the derived class.'

    def predict(self) -> np.array:
        return 'This is the implementation of the predict function from the derived class.'

if __name__ == '__main__':
    myDetector = subDetector()
    # myDetector.mkdir()
