import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from lib.classify import Classifier


class DocumentClassifier(Classifier):
    def __init__(self, confidence_threshold, enabled):
        super().__init__(confidence_threshold, name="document", allowed_classes=["handwritten", "presentation"])
        if enabled:
            self.processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
            self.model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _get_predicted_class(self, probabilities):
        # Sort predictions by confidence
        sorted_indices = torch.argsort(probabilities, descending=True)

        # Iterate over sorted predictions to find the first allowed class above the threshold
        for idx in sorted_indices:
            confidence = probabilities[idx].item()
            predicted_class = self.model.config.id2label[idx.item()]

            # Check if the class is allowed and meets the confidence threshold
            if (self.allowed_classes is None or predicted_class in self.allowed_classes) and confidence >= self.confidence_threshold:
                return predicted_class

        return None

    def classify(self, image_path):
        image = self._load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Calculate softmax probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return self._get_predicted_class(probabilities)


