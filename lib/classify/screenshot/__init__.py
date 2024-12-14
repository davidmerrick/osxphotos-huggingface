from lib.classify import PipelineClassifier

class ScreenshotClassifier(PipelineClassifier):
    def __init__(self, confidence_threshold, enabled):
        super().__init__(
            model_name="google/vit-base-patch16-224",
            confidence_threshold=confidence_threshold,
            name="screenshot",
            allowed_classes=["web site, website, internet site, site"],
            enabled=enabled
        )
