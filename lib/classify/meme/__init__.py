from lib.classify import PipelineClassifier


class MemeClassifier(PipelineClassifier):
    def __init__(self, confidence_threshold, enabled=True):
        super().__init__(
            model_name="davidmerrick/detect_meme",
            confidence_threshold=confidence_threshold,
            name="meme",
            allowed_classes=None,
            enabled=enabled
        )

    def _get_predicted_class(self, predictions):
        # Extract scores for "meme" and "non-meme"
        meme_score = next((pred['score'] for pred in predictions if pred['label'] == "meme"), 0)
        non_meme_score = next((pred['score'] for pred in predictions if pred['label'] == "non-meme"), 0)

        # Check both conditions
        if meme_score > non_meme_score and meme_score > self.confidence_threshold:
            return True  # Prediction is "meme"
        return False  # Otherwise, return "non-meme"
