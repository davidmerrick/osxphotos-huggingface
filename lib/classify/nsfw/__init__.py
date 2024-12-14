from lib.classify import PipelineClassifier

class NsfwClassifier(PipelineClassifier):
    def __init__(self, confidence_threshold, enabled):
        super().__init__(
            model_name="AdamCodd/vit-base-nsfw-detector",
            confidence_threshold=confidence_threshold,
            name="nsfw",
            allowed_classes=["nsfw"],
            enabled=enabled
        )
