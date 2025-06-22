from typing import Any

import cv2
import numpy as np
from abc import ABC, abstractmethod
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from keras.api.applications import ResNet50
from keras.api.applications.resnet50 import preprocess_input
from keras.api.models import Model
import tensorflow as tf
from sklearn.pipeline import make_pipeline, Pipeline


# ======= OpOption config =======
class OpOption:
    def __init__(self, dim_reduction: str, feature_extract: str, enabled=True):
        self.dim_reduction: str = dim_reduction
        self.feature_extraction: str = feature_extract
        self.enabled = enabled

    def __str__(self):
        return f"{self.dim_reduction.lower()}_{self.feature_extraction.lower()}"

    def get_chain(self):
        return [self.feature_extraction, self.dim_reduction]

    def to_dict(self):
        return {
            "dim_reduction": self.dim_reduction.lower(),
            "feature_extract": self.feature_extraction.lower()
        }


# ======= Strategy base =======
class PreprocessStrategy(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def prepare_gray_uint8(self, img):
        if img is None or img.size == 0:
            raise ValueError("Image is empty!")

        if len(img.shape) == 3:  # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.dtype != np.uint8:
            img = (img * 255).astype("uint8")

        return img

    def extract_features(self, img):
        pass


# ======= Feature Extractors =======

class HOGFeatureExtractor(PreprocessStrategy):
    def fit(self, X, y=None): pass

    def transform(self, X, y=None):
        from skimage.color import rgb2gray

        features = []
        for img in X:
            # Ensure grayscale: hog() now expects 2D
            if img.ndim == 3:
                img = rgb2gray(img)  # Converts (H, W, 3) → (H, W)
            features.append(
                hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            )
        return np.array(features)


class SIFTFeatureExtractor(PreprocessStrategy):
    def __init__(self, max_features=128):
        self.sift = cv2.SIFT_create()
        self.max_features = max_features

    def fit(self, X, y=None): pass

    def extract_features(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None:
            descriptors = np.zeros((1, 128), dtype=np.float32)
        # Flatten or average the descriptors to a fixed-length vector
        descriptors = descriptors[:self.max_features]
        flat_desc = descriptors.flatten()
        if len(flat_desc) < 128 * self.max_features:
            flat_desc = np.pad(flat_desc, (0, 128 * self.max_features - len(flat_desc)))
        return flat_desc

    def transform(self, X, y=None):

        features = []
        for i, img in enumerate(X):
            try:
                img_proc = self.prepare_gray_uint8(img)
                feats = []
                feats.extend(self.extract_features(img_proc))
                features.append(feats)

            except Exception as e:
                print(f"[WARN] Failed to process image {i}: {e}")
                features.append(np.zeros(12800))

        return np.array(features)


class GLCMFeatureExtractor(PreprocessStrategy):
    def __init__(self):
        self.distances = [1, 2]
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def _glcm_entropy(self, glcm):
        glcm_norm = glcm.astype(np.float32)
        glcm_norm /= (glcm_norm.sum() + 1e-12)
        return -np.sum(glcm_norm * np.log2(glcm_norm + 1e-12))

    def _glcm_cluster_features(self, glcm_2d):
        I, J = np.indices(glcm_2d.shape)
        mu_i = np.sum(I * glcm_2d)
        mu_j = np.sum(J * glcm_2d)
        cluster_shade = np.sum(((I + J - mu_i - mu_j) ** 3) * glcm_2d)
        cluster_prominence = np.sum(((I + J - mu_i - mu_j) ** 4) * glcm_2d)
        return cluster_shade, cluster_prominence

    def _byte_entropy(self, img):
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
        return -np.sum(hist * np.log2(hist + 1e-12))

    def fit(self, X, y=None): pass

    def extract_features(self, img):
        # Compute GLCM and extract texture features
        img = self.prepare_gray_uint8(img)

        glcm = graycomatrix(img,
                            distances=self.distances,
                            angles=self.angles,
                            levels=256,
                            symmetric=True,
                            normed=True)

        features = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'ASM').mean(),
            self._glcm_entropy(glcm),
            self._byte_entropy(img),
        ]

        # Collapse GLCM across angles to 2D for cluster metrics
        glcm_2d = glcm.sum(axis=(2, 3))
        cluster_shade, cluster_prominence = self._glcm_cluster_features(glcm_2d)
        features.extend([cluster_shade, cluster_prominence])

        return features

    def transform(self, X, y=None):
        features = []
        for i, img in enumerate(X):
            try:
                img_proc = self.prepare_gray_uint8(img)
                feats = []
                feats.extend(self.extract_features(img_proc))
                features.append(feats)

            except Exception as e:
                print(f"[WARN] Failed to process image {i}: {e}")
                features.append(np.zeros(6))

        return np.array(features)


class HistogramFeatureExtractor(PreprocessStrategy):
    def __init__(self, bins=32):
        self.bins = bins

    def fit(self, X, y=None): pass

    def transform(self, X, y=None):
        features = []
        for img in X:
            hist = cv2.calcHist([img], [0, 1, 2], None, [self.bins]*3, [0, 256]*3)
            features.append(cv2.normalize(hist, hist).flatten())
        return np.array(features)


class ResNetFeatureExtractor(PreprocessStrategy):
    def __init__(self):
        base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def fit(self, X, y=None): pass

    def transform(self, X, y=None):
        resized = np.array([tf.image.resize(img, (224, 224)) for img in X])
        prepped = preprocess_input(resized)
        return self.model.predict(prepped, verbose=0)


# ======= Dimensionality Reducers =======

class PCAProjector(PreprocessStrategy):
    def __init__(self, n_components=128):
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None): self.pca.fit(X)
    def transform(self, X, y=None): return self.pca.transform(X)


# ======= Just do nothing preprocessor =======

class DummyFeatureProcessor(PreprocessStrategy):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X
    def fit_transform(self, X, y=None): return X


# ======= Strategy Dispatcher =======

class FeatureProcessor:
    def __init__(self, option: OpOption):
        self.option = option
        self.strategy = None
        self._build_strategy()

    def _build_strategy(self):
        chain = self.option.get_chain()
        strategy_map = {
            "hog": HOGFeatureExtractor,
            "sift": SIFTFeatureExtractor,
            "glcm": GLCMFeatureExtractor,
            "hist": HistogramFeatureExtractor,
            "resnet": ResNetFeatureExtractor,
            "pca": PCAProjector,
            "raw": DummyFeatureProcessor
        }

        steps = []
        for name in chain:
            key = name.lower()
            if key not in strategy_map:
                raise ValueError(f"[!] Unsupported strategy in chain: {key}")
            steps.append(strategy_map[key]())

        if len(steps) == 1:
            self.strategy = steps[0]  # Return single instance
        self.strategy = make_pipeline(*steps)  # Compose pipeline

    def fit(self, X, y=None): self.strategy.fit(X)
    def transform(self, X, y=None): return self.strategy.transform(X)
    def fit_transform(self, X, y=None): return self.strategy.fit_transform(X)
