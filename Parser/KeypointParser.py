import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import skimage.draw

IMAGE_DIR = "images"


class KeypointParser:
    KEYPOINTS = {
        1: "R_Ankle",
        2: "R_Knee",
        3: "R_Hip",
        4: "L_Hip",
        5: "L_Knee",
        6: "L_Ankle",
        7: "B_Pelvis",
        8: "B_Spine",
        9: "B_Neck",
        10: "B_Head",
        11: "R_Wrist",
        12: "R_Elbow",
        13: "R_Shoulder",
        14: "L_Shoulder",
        15: "L_Elbow",
        16: "L_Wrist",
    }

    CONNECTED_JOINTS = {
        "right_lower_leg": (1, 2),
        "right_upper_leg": (2, 3),
        "hips": (3, 4),
        "left_upper_leg": (4, 5),
        "left_lower_leg": (5, 6),
        "spine": (7, 8),
        "shoulder": (13, 14),
        "right_upper_hand": (12, 13),
        "right_lower_hand": (11, 12),
        "left_upper_hand": (14, 15),
        "left_lower_hand": (15, 16),
        "neck": (9, 10),
    }

    COLORS = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (192, 192, 192),  # Silver
        (255, 165, 0),  # Orange
        (75, 0, 130),  # Indigo
        (240, 128, 128),  # Light Coral
        (0, 128, 128),  # Teal
    ]

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.empty_annotations = []

    @classmethod
    def get_keypoints(cls):
        return cls.KEYPOINTS.items()

    def get_empty_annotations(self):
        return self.empty_annotations

    def read_image_id_file(filepath):
        with open(filepath, "r") as file:
            data = file.readlines()
            return np.array([line.strip() for line in data])

    @staticmethod
    def get_all_annotations(annotation_dir, subset):
        """Load all annotations from the specified CSV file."""
        return pd.read_csv(
            os.path.join(annotation_dir, f"{subset}.csv"), header=None, index_col=0
        )

    @staticmethod
    def get_img_annotation(annotations, filename):
        return annotations.loc[filename]

    def get_image_dim(image_path):
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        return height, width

    # def process_image_to_keypoints(self, annotation, image_id, subset):
    #     filename = f"{image_id}"
    #     filename_no_extension = os.path.splitext(filename)[0]
    #     image_path = os.path.join(self.dataset_dir, IMAGE_DIR, subset, filename)

    #     if annotation is None:
    #         print(f"No annotations found for {filename_no_extension}.")
    #         self.empty_annotations.append(filename_no_extension)

    #     polygons = self.process_keypoints(keypoints=annotation)

    #     return image_path, polygons

    def process_keypoints_csv(self, annotations, scale_factor, image_size):
        """
        csv file line format:
        ImageID_PersonId.jpg,x1,y1,v1,x2,y2,v2,...x16,y16,v16
        Note: x,y, is the annotation label in (column, row),
        v stands for visuable
        """
        scale_factor_x, scale_factor_y = scale_factor
        image_size_x, image_size_y = image_size

        keypoints = []
        for i, rows in enumerate(annotations):
            curr_keypoints = []
            for j in range(0, len(rows), 3):
                x, y, _ = rows[j : j + 3]

                x = float(x) if not np.isnan(x) else -1.0
                y = float(y) if not np.isnan(y) else -1.0

                scaled_x = x * scale_factor_x[i]
                scaled_y = y * scale_factor_y[i]

                normalized_x = scaled_x / image_size_x
                normalized_y = scaled_y / image_size_y

                curr_keypoints.append((normalized_x, normalized_y))
            keypoints.append(curr_keypoints)

        return keypoints

    def process_keypoints(self, keypoints, scale_factor, image_size):
        polygons = []

        scale_factor_x, scale_factor_y = scale_factor
        image_size_x, image_size_y = image_size

        for i in range(0, len(keypoints), 3):
            x, y, _ = keypoints[i : i + 3]

            x = float(x) if not np.isnan(x) else -1.0
            y = float(y) if not np.isnan(y) else -1.0

            # Scale the keypoints
            scaled_x = x * scale_factor_x
            scaled_y = y * scale_factor_y

            normalized_x = scaled_x / image_size_x
            normalized_y = scaled_y / image_size_y

            polygons.append((normalized_x, normalized_y))

        return polygons

    def draw_keypoints_and_connections(self, image, keypoints):
        # Load the image
        if isinstance(image, np.ndarray):
            print("Image shape:", image.shape)  # Check the shape of the image
            image = image.astype("uint8")

        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            # Draw the keypoint

            if x > 0 or y > 0:
                cv2.circle(image, (int(x), int(y)), 5, self.COLORS[i], -1)
                cv2.putText(
                    image,
                    self.KEYPOINTS[i + 1],
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLORS[i],
                    1,
                )

        # Draw connections
        for connection in self.CONNECTED_JOINTS.values():
            start_idx, end_idx = connection
            if start_idx <= len(keypoints) and end_idx <= len(keypoints):
                x_start, y_start = keypoints[start_idx - 1]
                x_end, y_end = keypoints[end_idx - 1]

                x_start = np.nan_to_num(x_start, nan=0)
                y_start = np.nan_to_num(y_start, nan=0)
                x_end = np.nan_to_num(x_end, nan=0)
                y_end = np.nan_to_num(y_end, nan=0)

                if x_start > 0 or y > 0 or x > 0 or y_end > 0:
                    cv2.line(
                        image,
                        (int(x_start), int(y_start)),
                        (int(x_end), int(y_end)),
                        (255, 255, 255),
                        2,
                    )

        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    def denormalize_image(image):
        image_np = image.numpy()
        return image_np * 255.0

    def denormalize_annotation(annotation, image_size):
        anno_np = annotation.numpy()
        return anno_np * image_size


def process_sample(parser, image_id, subset, annotations, image):
    """Process a single image sample by its ID."""
    # Get the annotation for the specified image ID
    annotation = parser.get_img_annotation(annotations, image_id)

    # Process the image and keypoints
    image_path, polygons = parser.process_image_to_keypoints(
        annotation.values, image_id, subset
    )

    print(image_path)

    target_size = 256

    h, w = image.shape[:2]

    resized_image = cv2.resize(image, (target_size, target_size))

    x_scale_factor = target_size / w
    y_scale_factor = target_size / h

    print(x_scale_factor, y_scale_factor)

    reformed_polygons = []

    for x, y in polygons:
        new_x = x_scale_factor * x
        new_y = y_scale_factor * y
        reformed_polygons.append((new_x, new_y))

    print(reformed_polygons)

    print(h, w)

    # Visualize the keypoints on the image
    parser.draw_keypoints_and_connections(resized_image, reformed_polygons)
