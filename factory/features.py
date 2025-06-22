from sklearn.pipeline import make_pipeline, Pipeline
from typing import Type, Any
from core.features import (
    HOGFeatureExtractor,
    SIFTFeatureExtractor,
    GLCMFeatureExtractor,
    HistogramFeatureExtractor,
    ResNetFeatureExtractor,
    PCAProjector,
    DummyFeatureProcessor,
    PreprocessStrategy,
    OpOption
)


class FeatureProcessorFactory:
    _strategy_map: dict[str, Type[PreprocessStrategy]] = {
        "hog": HOGFeatureExtractor,
        "sift": SIFTFeatureExtractor,
        "glcm": GLCMFeatureExtractor,
        "hist": HistogramFeatureExtractor,
        "resnet": ResNetFeatureExtractor,
        "pca": PCAProjector,
        "raw": DummyFeatureProcessor
    }

    @classmethod
    def build(cls, option: OpOption) -> Pipeline | Any:
        chain = option.get_chain()
        steps: list[PreprocessStrategy] = []

        for name in chain:
            key = name.strip().lower()
            if key not in cls._strategy_map:
                raise ValueError(f"[!] Unsupported strategy: {key}")
            steps.append(cls._strategy_map[key]())

        if len(steps) == 1:
            return steps[0]
        return make_pipeline(*steps)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._strategy_map.keys())

    @classmethod
    def list_all_dim_reduction(cls) -> list[str]:
        return ["raw", "pca"]

    @classmethod
    def list_all_feature_extract(cls) -> list[str]:
        fe = cls.list_all()
        for dr in cls.list_all_dim_reduction():
            fe.remove(dr)
        fe.append("raw")
        return fe


