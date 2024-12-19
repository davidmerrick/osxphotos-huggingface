import cv2

from lib.classify import Classifier


class BarcodeClassifier(Classifier):
    def __init__(self, confidence_threshold, enabled=True):
        super().__init__(
            confidence_threshold,
            name="barcode",
            allowed_classes=None,
            enabled=enabled
        )

    def classify(self, image_path):
        img = cv2.imread(image_path)
        barcode_detector = cv2.barcode.BarcodeDetector()

        # 'retval' is boolean mentioning whether barcode has been detected or not
        if barcode_detector.detect(img)[0]:
           return True
        return False
