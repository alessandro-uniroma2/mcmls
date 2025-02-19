import importlib
import os
from typing import Union

from models.imodel import IModel
from utils.common import get_project_root
from utils.stringutils import camel_to_snake, snake_to_camel


class ModelFactory:
    @staticmethod
    def from_name(name: str, **kwargs) -> Union[IModel, None]:
        try:
            module = importlib.import_module(f"models.{camel_to_snake(name)}")
            _class_name = name
            _class = getattr(module, f"{_class_name}Model")
            return _class(**kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def list_all():
        path = str(get_project_root().joinpath("mcmls", "models").absolute())
        models = [
            f.replace(".py", "") for f in os.listdir(path) if
            os.path.isfile(os.path.join(path, f))
            and f not in [
                "__init__.py",
                "imodel.py",
                ".gitignore"
            ]
        ]
        return [snake_to_camel(m) for m in models]
