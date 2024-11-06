import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class SegmentationParser:
    """Segmentation for Look Into Person (LIP) dataset"""

    # Class definitions for LIP dataset
    parsing_classes = {
        0: "Background",
        1: "Hat",
        2: "Hair",
        3: "Glove",
        4: "Sunglasses",
        5: "UpperClothes",
        6: "Dress",
        7: "Coat",
        8: "Socks",
        9: "Pants",
        10: "Jumpsuits",
        11: "Scarf",
        12: "Skirt",
        13: "Face",
        14: "Left-arm",
        15: "Right-arm",
        16: "Left-leg",
        17: "Right-leg",
        18: "Left-shoe",
        19: "Right-shoe",
    }

    def __init__(self, dataset_dir):
        """
        Initialize LIP parser

        Args:
            dataset_dir: Root directory of LIP dataset
        """
        self.dataset_dir = dataset_dir
        self.colors = self._generate_colors()

    def _generate_colors(self):
        """Generate distinct colors for visualization"""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.parsing_classes), 3))
        # Set background color to black
        colors[0] = [0, 0, 0]
        return colors

    # ADDED TRAIN path
    def load_image(self, image_id):
        """Load image by ID"""
        image_path = os.path.join(
            self.dataset_dir, "images", "train", f"{image_id}.jpg"
        )
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_parsing_mask(self, image_id):
        """
        Load parsing mask for given image ID

        Args:
            image_id: Image identifier without extension

        Returns:
            numpy array: Parsing mask with class IDs
        """
        mask_path = os.path.join(self.dataset_dir, "masks", "train", f"{image_id}.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Use PIL to ensure proper parsing of the mask
        mask = np.array(Image.open(mask_path))
        return mask

    def create_binary_masks(self, parsing_mask):
        """
        Convert parsing mask to binary instance masks

        Args:
            parsing_mask: Single mask with class IDs

        Returns:
            tuple: (masks, class_ids)
                - masks: Boolean array [height, width, instance_count]
                - class_ids: Array of class IDs
        """
        height, width = parsing_mask.shape
        unique_classes = np.unique(parsing_mask)
        # Remove background class (0)
        unique_classes = unique_classes[unique_classes != 0]

        masks = np.zeros((height, width, len(unique_classes)), dtype=bool)
        class_ids = np.zeros(len(unique_classes), dtype=np.int32)

        for idx, class_id in enumerate(unique_classes):
            masks[:, :, idx] = parsing_mask == class_id
            class_ids[idx] = class_id

        return masks, class_ids

    def visualize_parsing(self, image, parsing_mask, alpha=0.5):
        """
        Create visualization of parsing mask overlaid on image

        Args:
            image: Original RGB image
            parsing_mask: Parsing annotation mask
            alpha: Transparency of overlay

        Returns:
            numpy array: Visualization image
        """
        vis_image = image.copy()

        # Create colored overlay
        overlay = np.zeros_like(image)
        for class_id in range(len(self.parsing_classes)):
            mask = parsing_mask == class_id
            if mask.any():
                color = self.colors[class_id]
                overlay[mask] = color

        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
        return vis_image

    def get_class_masks(self, parsing_mask):
        """
        Get individual masks for each class

        Args:
            parsing_mask: Full parsing mask

        Returns:
            dict: Mapping of class names to binary masks
        """
        class_masks = {}
        for class_id, class_name in self.parsing_classes.items():
            mask = parsing_mask == class_id
            if mask.any():
                class_masks[class_name] = mask
        return class_masks

    def visualize_all_classes(self, image_id):
        """
        Create visualization of all classes separately

        Args:
            image_id: Image identifier

        Returns:
            matplotlib figure
        """
        image = self.load_image(image_id)
        parsing_mask = self.load_parsing_mask(image_id)
        class_masks = self.get_class_masks(parsing_mask)

        # Create grid plot
        n_classes = len(class_masks)
        n_cols = 4
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        # Plot each class mask
        for idx, (class_name, mask) in enumerate(class_masks.items()):
            if idx < len(axes):
                vis = image.copy()
                vis[~mask] = vis[~mask] // 4  # Dim non-mask regions
                axes[idx].imshow(vis)
                axes[idx].set_title(class_name)
                axes[idx].axis("off")

        # Clear unused subplots
        for idx in range(len(class_masks), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        return fig


def process_sample(parser, image_id):
    """
    Process a single sample from the dataset

    Args:
        parser: LIPParser instance
        image_id: Image identifier

    Returns:
        dict: Processed data including image, masks, and visualizations
    """
    # Load data
    image = parser.load_image(image_id)
    parsing_mask = parser.load_parsing_mask(image_id)

    # Create binary instance masks
    masks, class_ids = parser.create_binary_masks(parsing_mask)

    # Create visualization
    vis_image = parser.visualize_parsing(image, parsing_mask)

    return {
        "image": image,
        "parsing_mask": parsing_mask,
        "instance_masks": masks,
        "class_ids": class_ids,
        "visualization": vis_image,
    }
