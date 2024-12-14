from typing import List

import Quartz
import objc
from Cocoa import NSURL
from Foundation import NSDictionary

from lib.classify import Classifier


class QRClassifier(Classifier):
    def __init__(self, confidence_threshold, enabled=True):
        super().__init__(confidence_threshold, name="qr", allowed_classes=None, enabled=enabled)

    def classify(self, image_path):
        return self._find_all_qrcodes(image_path) != []

    def _find_all_qrcodes(self, image_path: str) -> List[str]:
        """Detect QR Codes in images using CIDetector and return text of the found QR Codes"""
        with objc.autorelease_pool():
            context = Quartz.CIContext.contextWithOptions_(None)
            options = NSDictionary.dictionaryWithDictionary_(
                {"CIDetectorAccuracy": Quartz.CIDetectorAccuracyHigh}
            )
            detector = Quartz.CIDetector.detectorOfType_context_options_(
                Quartz.CIDetectorTypeQRCode, context, options
            )

            results = []
            input_url = NSURL.fileURLWithPath_(image_path)
            input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)
            features = detector.featuresInImage_(input_image)

            if not features:
                return []
            for idx in range(features.count()):
                feature = features.objectAtIndex_(idx)
                results.append(feature.messageString())
            return results
