import random

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, AdamW, AutoImageProcessor

from lib.osxphotos_utils import construct_query_options
from lib.osxphotos_utils.photoprocessor import PhotoProcessor


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ModelTuner:
    """
    A very osxphotos-specific model fine-tuner.
    Allows you to create albums for training in Apple Photos directly and pass a mapping of labels to these albums.
    """
    def __init__(self, verbose_mode, library_path, output_path="./classifier"):
        self.processor = PhotoProcessor(
            keystore_name="training",
            verbose_mode=verbose_mode,
            library_path=library_path
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.output_path = output_path
        self.model = None
        self.optimizer = None
        self.criterion = CrossEntropyLoss()

    def _get_preview_paths(self, album_name: str, dry_run: bool):
        ctxs = self.processor.get_contexts(construct_query_options(album=[album_name]), dry_run)
        return [ctx.preview_path for ctx in ctxs]

    def _validate_model(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images).logits
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {correct / total * 100:.2f}%")

    def _prepare_datasets(self, label_album_mapping, dry_run):
        labeled_data = []
        label_mapping = {label: idx for idx, (label, _) in enumerate(label_album_mapping)}

        # Collect and label data
        for label, album_name in label_album_mapping:
            paths = self._get_preview_paths(album_name, dry_run)
            labeled_data.extend([(path, label_mapping[label]) for path in paths])

        random.shuffle(labeled_data)

        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
        train_dataset = CustomDataset(train_data, transform=transform)
        val_dataset = CustomDataset(val_data, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return train_loader, val_loader, label_mapping

    def train(self, label_album_mapping, dry_run=False, epochs=5):
        train_loader, val_loader, label_mapping = self._prepare_datasets(label_album_mapping, dry_run)

        num_labels = len(label_mapping)

        # Initialize model with the correct number of labels
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            id2label={v: k for k, v in label_mapping.items()},
            label2id=label_mapping,
            ignore_mismatched_sizes=True
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

            # Validate the model
            self._validate_model(val_loader)

        # Save the model and processor
        self.model.save_pretrained(self.output_path)
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        processor.save_pretrained(self.output_path)

        print(f"Model and processor saved to {self.output_path}")
