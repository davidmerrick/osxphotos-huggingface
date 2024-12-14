from abc import abstractmethod, ABC

from PIL import Image
from transformers import pipeline


class Classifier(ABC):
    def __init__(
        self,
        confidence_threshold,
        name: str,
        allowed_classes=None,
        enabled=True
    ):
        self.confidence_threshold = confidence_threshold
        self.name = name
        self.allowed_classes = allowed_classes
        self.enabled = enabled

    @abstractmethod
    def classify(self, image_path):
        pass


class PipelineClassifier(Classifier):
    def __init__(
        self,
        model_name: str,
        confidence_threshold,
        name: str,
        allowed_classes=None,
        enabled=True
    ):
        super().__init__(
            confidence_threshold=confidence_threshold,
            name=name,
            allowed_classes=allowed_classes,
            enabled=enabled
        )
        if enabled:
            self.pipeline = pipeline("image-classification", model=model_name, use_fast=True)

    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image.verify()
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def classify(self, image_path):
        if not self.enabled:
            raise ValueError("Classifier is not enabled")

        image = self._load_image(image_path)
        if image is None:
            return None
        predictions = self.pipeline(image)  # Pass the image directly to the pipeline
        return self._get_predicted_class(predictions)

    def _get_predicted_class(self, predictions):
        score = next((pred['score'] for pred in predictions if pred['label'] in self.allowed_classes), 0)
        return score > self.confidence_threshold
