# chess_computer_vision
Deep seek : 



Of course. This third paper, "Determining Chess Game State From an Image," presents a highly accurate, vision-only system for reconstructing the entire board state from a single image. Its major innovation is a state-of-the-art accuracy and a novel method to adapt to new chess sets with minimal data.

Here is a summary of the steps for this system.

System Overview

This system is designed for a static, single image (not a video or robot). It takes a photo of a chessboard from an angle (not necessarily a top-down view) and outputs the complete game state in Forsyth–Edwards Notation (FEN). Its pipeline is a three-stage, end-to-end process that combines traditional computer vision with deep learning.

Step-by-Step Summary

Step 1: Board Localisation

The goal is to find the four corner points of the chessboard in the image to correct for perspective distortion.

Process:

Edge & Line Detection: The input image is converted to grayscale. The Canny edge detector finds edges, and the Hough transform detects lines.
Cluster Lines: Detected lines are clustered into "horizontal" and "vertical" groups using an agglomerative clustering algorithm, which is more robust to severe camera tilts than simple angle thresholds.
Find Intersections: Intersection points of these horizontal and vertical lines are computed.
Find the Chess Grid (Homography): A RANSAC-based algorithm is used to find the best projective transformation (homography matrix) that maps the detected intersection points onto a regular grid. This algorithm intelligently figures out how many squares are between the detected lines.
Recover Missing Lines: If not all 9x9 grid lines are found, the system analyzes gradient intensities in the warped image to deduce where the missing outer lines should be.
Output: A precise homography matrix that can warp the input image into a perfect, top-down view of the chessboard.
Step 2: Occupancy Classification

Before identifying the pieces, the system determines which squares are empty and which are occupied. This is a crucial step to reduce false positives.

Process:

Warp and Crop: The input image is warped using the homography from Step 1. Each square is cropped from this corrected image. The bounding box is increased by 50% to include contextual information from adjacent squares, which proves vital for accuracy.
Binary Classification: A binary CNN classifier (a fine-tuned ResNet model) is trained to classify each cropped square as either "occupied" or "empty."
Output: A binary mask of the board indicating occupied squares.
Step 3: Piece Classification

This step identifies the specific type and color of the piece on each occupied square.

Process:

Smart Cropping: Pieces are cropped with a variable bounding box. The box is made taller for pieces at the back of the board to include distinctive features (e.g., a queen's crown). Pieces on the left side of the board are horizontally flipped so the square of interest is always in the bottom left, providing a consistent frame of reference for the classifier.
Multi-Class Classification: A multi-class CNN classifier (a fine-tuned InceptionV3 model) is trained to classify the cropped image into one of 13 classes (6 piece types for both black and white, plus a redundant "empty" class).
Output: The complete board state, which is then converted into a FEN string.
Key Innovation: Few-Shot Transfer Learning to Unseen Chess Sets

A major challenge is that a model trained on one chess set will perform poorly on another with a different design. This paper introduces a powerful solution requiring only two unlabeled images.

Process:

Data Collection: Take two photos of the new, unseen chess set: one from the white player's perspective and one from the black player's perspective, both in the standard starting position.
Automatic Labeling: Since the position (the starting position) is known, the system can automatically generate labeled training data by running the board localisation algorithm on these two images. This provides 128 samples for the occupancy classifier (64 squares × 2 images) and 64 samples for the piece classifier (32 pieces × 2 images).
Fine-Tuning with Heavy Augmentation: The pre-trained CNNs from the main system are fine-tuned on this small, new dataset. For the piece classifier, heavy data augmentation (shearing, color jitter, scaling, translation) is critical to prevent overfitting and teach the model the new pieces' appearance from various perspectives.
Performance and Results

This system achieved groundbreaking accuracy:

Per-square error rate: 0.23% on the test set, which is 28 times better than the previous state-of-the-art.
End-to-end board accuracy: 93.86% of boards were perfectly recognized, and 99.71% had at most one error.
Adaptation Success: After fine-tuning with just two images, the system achieved a per-square accuracy of 99.83% on the new, real-world chess set.
Summary of the Workflow

Input: A single image of a chessboard from an angle.
Localise: Find the board's corners and warp the image to a top-down view.
Check Occupancy: For each square, use a CNN to decide if a piece is present.
Identify Pieces: For each occupied square, use another CNN to identify the specific piece and its color.
Output: A FEN string representing the complete game state.
In essence, this paper presents a highly accurate, modular, and adaptable pipeline that solves the chess recognition problem for a single image with state-of-the-art performance and a practical method for generalizing to new chess sets.




