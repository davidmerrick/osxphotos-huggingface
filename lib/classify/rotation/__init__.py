import logging

from lib.classify import Classifier

# Set the logging level for timm to WARNING or ERROR
logging.getLogger("timm").setLevel(logging.WARNING)

import torch.nn.parallel
import torch.utils.data

import numpy as np
from timm import create_model

import torch
import torch.nn.parallel
import torch.utils.data
import yaml
from albumentations.core.serialization import from_dict
from huggingface_hub import hf_hub_download
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from torch import nn

class RotatedClassifier(Classifier):
    def __init__(
        self,
        confidence_threshold,
        enabled=True
    ):
        super().__init__(
            confidence_threshold,
            name="rotated",
            allowed_classes=None,
            enabled=enabled
        )

        if not enabled:
            pass

        repo_id = "davidmerrick/detect_rotated"  # Your Hugging Face repository ID
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        weight_path = hf_hub_download(repo_id=repo_id, filename="model.pth")

        self.fp16 = True
        with open(config_path) as f:
            hparams = yaml.safe_load(f)

        hparams.update({"fp16": self.fp16})

        # Select device: MPS or CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Initialize and load the model
        model = object_from_dict(hparams["model"])

        # Load the state dictionary
        state_dict = torch.load(weight_path, map_location="cpu")

        # Extract "state_dict" if it's nested
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip "model." prefix from keys if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Load weights into the model
        model.load_state_dict(state_dict)

        model = nn.Sequential(model, nn.Softmax(dim=1))
        model = model.to(self.device)
        if self.fp16:
            model = model.half()

        self.transform = from_dict(hparams["test_aug"])
        self.model = model

    def classify(self, image_path):
        # Load and preprocess the single image
        image_path = image_path

        image_path = load_rgb(image_path)
        image_path = self.transform(image=image_path)["image"]
        torched_image = tensor_from_rgb_image(image_path).unsqueeze(0).to(self.device)  # Add batch dimension

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            if self.fp16:
                torched_image = torched_image.half()

            prediction = self.model(torched_image).squeeze(0).cpu().numpy()  # Remove batch dimension

            # Return the angle with the highest confidence
            return self._get_highest_confidence_angle(prediction)

    def _get_highest_confidence_angle(self, prediction):
        """
        Given a numpy array of confidence values for rotation angles [0º, 90º, 180º, 270º],
        return the angle corresponding to the highest confidence as an integer.
        If no confidence values exceed the threshold, log a debug message and return 0.

        Args:
            prediction (numpy.ndarray): Array of confidence values for angles [0, 90, 180, 270].
            confidence_threshold (float): Minimum confidence value to consider a valid prediction.

        Returns:
            int: The angle with the highest confidence, or None
        """
        angles = [0, 90, 180, 270]
        max_index = np.argmax(prediction)
        max_confidence = prediction[max_index]

        if max_confidence >= self.confidence_threshold:
            if(angles[max_index] > 0):
                return angles[max_index]
            else:
                return None
        else:
            logging.debug(f"No confidence values exceed the threshold of {self.confidence_threshold}. Defaulting to 0.")
            return None


    def _load_model_with_weights(self, model_name, num_classes, pretrained, model_path):
        """
        Load a timm model and correctly map the weights from the given state dictionary.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes for the model.
            pretrained (bool): Whether to load pretrained weights.
            model_path (str): Path to the saved weights.

        Returns:
            torch.nn.Module: The loaded model.
        """
        # Initialize the model
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle nested `state_dict` if present
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip "model." prefix from keys if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Load the weights into the model
        model.load_state_dict(state_dict)
        return model
