import configparser
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import splitfolders

from core.datatypes import MalwareImages
from utils.common import get_project_root


class DatasetLoader:
    def __init__(self):
        # Define data directories using pathlib for OS-independent paths
        self.project_root = get_project_root()
        print(f"(*) Using project root: {self.project_root}")
        config_folder = self.project_root.joinpath("mcmls", "config")
        cf = config_folder.joinpath("config.ini")
        self.config = configparser.ConfigParser(allow_no_value=True, interpolation=configparser.ExtendedInterpolation())
        self.config.read(str(cf))
        self.project_root = self.project_root.joinpath("vitaminc")
        dataset_cf = self.project_root.joinpath(self.config.get("DATASET", "config")).resolve()
        self.dataset_config = {}
        self.training_datasets: Dict[str, MalwareImages] = dict()
        # This seed is l33t
        self.seed = 1337
        with open(str(dataset_cf), "r") as _in:
            datasets = json.load(_in)
            for dataset in datasets.get("datasets"):
                self.dataset_config[dataset.get("name")] = dataset

        self.__base_data_directory = self.project_root.joinpath("data")
        self.__base_split_directory = self.project_root.joinpath("split")

        # Ensure existence
        self.__base_data_directory.mkdir(exist_ok=True)
        self.__base_split_directory.mkdir(exist_ok=True)

    def list_datasets(self) -> List[str]:
        if not self.dataset_config:
            return []
        return [x for x in self.dataset_config.keys()]

    def get_dataset_directory(self, dataset_name: str) -> Path:
        # Dataset directories
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        return self.__base_data_directory.joinpath(f"{dataset_name}_dataset")

    def get_dataset_image_size(self, dataset_name: str) -> Tuple[int, int]:
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        w = int(self.dataset_config.get(dataset_name, {}).get("image_size").get("w"))
        h = int(self.dataset_config.get(dataset_name, {}).get("image_size").get("h"))
        return w, h

    def get_dataset_image_colormode(self, dataset_name: str) -> str:
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        mode = self.dataset_config.get(dataset_name, {}).get("colormode")
        if mode not in ["rgb", "grayscale"]:
            return "rgb"
        return mode

    def get_dataset_class_number(self, dataset_name: str) -> int:
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        classes = self.dataset_config.get(dataset_name, {}).get("classes")
        return int(classes)

    def get_dataset_options(self, dataset_name: str) -> dict:
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        options = self.dataset_config.get(dataset_name, {}).get("options")
        return options

    def get_split_directory(self, dataset_name: str) -> Path:
        # Split Dataset directories
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        return self.__base_split_directory.joinpath(f"{dataset_name}_dataset")

    # Splitting the dataset into training and testing subsets proportionally.
    def __split_dataset(self, src: str, dst: str, train_ratio: float, test_ratio: float):
        splitfolders.ratio(input=src,
                           output=dst,
                           seed=self.seed,
                           ratio=(train_ratio, 0, test_ratio),
                           group_prefix=None,
                           move=False)

    def split_dataset(self, dataset_name: str) -> bool:
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        train_d = self.get_dataset_directory(dataset_name).joinpath("train")
        new_train_test_d = self.get_split_directory(dataset_name)
        # Ensure the split doesn't exist yet
        if not new_train_test_d.is_dir():
            self.__split_dataset(str(train_d), str(new_train_test_d), 0.8, 0.2)

        # Cheating: just check for the directories to exist
        return self.get_training_data_dir(dataset_name).is_dir() and self.get_testing_data_dir(dataset_name).is_dir()

    def get_training_data_dir(self, dataset_name: str, split: bool = True) -> Path:
        if split:
            return self.get_split_directory(dataset_name).joinpath("train")
        return self.get_dataset_directory(dataset_name).joinpath("train")

    def get_validation_data_dir(self, dataset_name: str) -> Path:
        return self.get_dataset_directory(dataset_name).joinpath("val")

    def get_testing_data_dir(self, dataset_name: str) -> Path:
        return self.get_split_directory(dataset_name).joinpath("test")

    def split_all_dataset(self) -> bool:
        success = []
        for dataset_name in self.list_datasets():
            print(f"(*) Splitting dataset {dataset_name} in Train / Test / Validate")
            success.append(self.split_dataset(dataset_name))
        return all(success)

    def load_dataset(self, dataset_name: str, split: bool):
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        self.training_datasets[dataset_name] = MalwareImages(
            data_dir=str(self.get_training_data_dir(dataset_name, split=split)),
            n=int(self.dataset_config.get(dataset_name, {}).get("d"))
        )

    def load_datasets(self, split: bool):
        for dataset_name in self.list_datasets():
            self.load_dataset(dataset_name, split=split)

    def plot_datasets(self):
        for dataset_name in self.list_datasets():
            self.plot_dataset(dataset_name)

    def plot_dataset(self, dataset_name: str):
        if dataset_name not in self.list_datasets():
            raise ValueError(f"Dataset {dataset_name} not supported")
        dataset = self.training_datasets.get(dataset_name)
        if dataset is not None:
            dataset.plot_class_distribution()
            dataset.malware_samples()
