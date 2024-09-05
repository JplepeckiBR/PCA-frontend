
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def extract_average_colors(image_path: Image.Image):
    """
    Extract average colors for skin, brows, hair, lips, and eyes from an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: A tuple containing the average RGB values for skin, brows, hair, lips, and eyes.
    """

    # Determine device for computation
    device = (
        "cuda"  # NVIDIA or AMD GPUs
        if torch.cuda.is_available()
        else "mps"  # Apple Silicon (Metal Performance Shaders)
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load models
    image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)

    # Load image
    image = Image.open(image_path)

    # Run inference on image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # Resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # H x W
        mode='bilinear',
        align_corners=False
    )

    # Get label masks
    labels = upsampled_logits.argmax(dim=1)[0]

    # Move to CPU for further processing
    labels_viz = labels.cpu().numpy()
    plt.imshow(labels_viz)
    plt.show()

    plt.imshow(image)

    # Extract RGB channels
    R = np.array(image)[:, :, 0]
    G = np.array(image)[:, :, 1]
    B = np.array(image)[:, :, 2]

    # Define segments
    def segment_color(foreground, label_value):
        return np.dstack((
            R * (foreground == label_value),
            G * (foreground == label_value),
            B * (foreground == label_value)
        ))

    skin = segment_color(labels_viz, 1)
    hair = segment_color(labels_viz, 13)
    left_brow = segment_color(labels_viz, 6)
    right_brow = segment_color(labels_viz, 7)
    brows = left_brow + right_brow
    left_eye = segment_color(labels_viz, 4)
    right_eye = segment_color(labels_viz, 5)
    eyes = left_eye + right_eye
    lip1 = segment_color(labels_viz, 12)
    lip2 = segment_color(labels_viz, 11)
    lips = lip1 + lip2

    # Calculate average colors
    def average_color(segment):
        flat_segment = segment.reshape(-1, 3)
        non_zero_pixels = flat_segment[np.any(flat_segment != 0, axis=1)]
        if len(non_zero_pixels) == 0:
            return (0, 0, 0)
        mean_red = np.mean(non_zero_pixels[:, 0])
        mean_green = np.mean(non_zero_pixels[:, 1])
        mean_blue = np.mean(non_zero_pixels[:, 2])
        return (int(mean_red), int(mean_green), int(mean_blue))

    average_skin_color = average_color(skin)
    average_brows_color = average_color(brows)
    average_hair_color = average_color(hair)
    average_lip_color = average_color(lips)
    average_eye_color = average_color(eyes)

    colors = [
        (average_skin_color, "Skin"),
        (average_brows_color, "Brows"),
        (average_hair_color, "Hair"),
        (average_eye_color, "Eyes"),
        (average_lip_color, "Lips")
    ]

    fig, axs = plt.subplots(1, len(colors), figsize=(15, 3))

    for ax, (color, title) in zip(axs, colors):
        image = np.ones((100, 100, 3))
        image[:, :, 0] *= color[0] / 255.0
        image[:, :, 1] *= color[1] / 255.0
        image[:, :, 2] *= color[2] / 255.0
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

    plt.show()

    return average_skin_color, average_brows_color, average_hair_color, average_lip_color, average_eye_color

# Example usage:
image_path = "raw_data/images of people i know/test2.jpg"
average_skin_color, average_brows_color, average_hair_color, average_lip_color, average_eye_color = extract_average_colors(image_path)





def predict_skin_tone_classification(average_skin_color, data=None):
    """
    Predicts the classification (season label) for a given average skin color using K-Nearest Neighbors.

    Parameters:
    average_skin_color (list or tuple): A list or tuple representing the RGB values of the average skin color.
    data (list of tuples, optional): A list of color tuples where each tuple is of the form (R, G, B, 'Label').

    Returns:
    str: The predicted classification (season label) for the provided average skin color.
    """
    # Static variable to store the trained model and scaler
    if not hasattr(predict_skin_tone_classification, "model"):
        # Use the provided data or a default dataset if none is provided
        if data is None:
            data = [
    (245, 225, 205, 'Warm'),
    (230, 210, 170, 'Warm'),
    (210, 180, 130, 'Warm'),
    (180, 140, 90, 'Warm'),
    (130, 95, 50, 'Warm'),
    (230, 210, 180, 'Warm'),
    (245, 225, 200, 'Warm'),
    (220, 190, 160, 'Warm'),
    (200, 170, 140, 'Warm'),
    (175, 135, 95, 'Warm'),
    (145, 100, 60, 'Warm'),
    (110, 75, 45, 'Warm'),

    (255, 220, 180, 'Warm'),  # Very light warm
    (240, 200, 160, 'Warm'),  # Light-medium warm
    (225, 185, 140, 'Warm'),  # Medium warm
    (190, 150, 110, 'Warm'),  # Medium-dark warm
    (170, 120, 80, 'Warm'),   # Dark warm
    (140, 95, 60, 'Warm'),    # Very dark warm
    (255, 235, 210, 'Warm'),  # Lightest warm
    (235, 195, 155, 'Warm'),  # Medium warm
    (210, 160, 120, 'Warm'),  # Medium-dark warm
    (185, 130, 90, 'Warm'),   # Dark warm
    (160, 105, 70, 'Warm'),   # Darker warm
    (135, 85, 55, 'Warm'),    # Very dark warm

    (255, 250, 235, 'Warm'),  # Very light warm
    (240, 220, 195, 'Warm'),  # Light-medium warm
    (225, 200, 165, 'Warm'),  # Medium warm
    (200, 175, 140, 'Warm'),  # Medium-dark warm
    (180, 150, 115, 'Warm'),  # Dark warm
    (150, 120, 85, 'Warm'),   # Very dark warm
    (255, 245, 220, 'Warm'),  # Lightest warm
    (245, 210, 175, 'Warm'),  # Medium warm
    (225, 185, 150, 'Warm'),  # Medium-dark warm
    (200, 160, 125, 'Warm'),  # Dark warm
    (175, 140, 100, 'Warm'),  # Darker warm
    (150, 115, 80, 'Warm'),   # Very dark warm
    (255, 240, 200, 'Warm'),  # Light warm
    (240, 210, 175, 'Warm'),  # Light-medium warm
    (225, 190, 155, 'Warm'),  # Medium warm
    (200, 160, 125, 'Warm'),  # Medium-dark warm
    (180, 140, 100, 'Warm'),  # Dark warm
    (150, 110, 75, 'Warm'),   # Darker warm
    (130, 95, 65, 'Warm'),    # Very dark warm
    (245, 225, 190, 'Warm'),  # Lightest warm

    (255, 245, 220, 'Warm'),  # Warm Ivory
    (250, 230, 190, 'Warm'),  # Warm Cream
    (245, 215, 170, 'Warm'),  # Warm Beige
    (235, 200, 145, 'Warm'),  # Warm Sand
    (220, 185, 130, 'Warm'),  # Warm Taupe
    (210, 175, 115, 'Warm'),  # Warm Caramel
    (195, 160, 105, 'Warm'),  # Warm Chestnut
    (180, 145, 95, 'Warm'),   # Warm Cinnamon
    (165, 130, 85, 'Warm'),   # Warm Rust
    (150, 115, 75, 'Warm'),   # Warm Sienna
    (135, 100, 65, 'Warm'),   # Warm Ochre
    (120, 85, 55, 'Warm'),    # Warm Clay
    (110, 75, 50, 'Warm'),    # Warm Walnut
    (95, 65, 45, 'Warm'),     # Warm Cocoa
    (80, 55, 35, 'Warm'),     # Warm Espresso
    (70, 45, 30, 'Warm'),     # Warm Mahogany
    (60, 40, 25, 'Warm'),     # Warm Coffee
    (50, 35, 20, 'Warm'),     # Warm Mocha
    (45, 30, 15, 'Warm'),     # Warm Toffee
    (35, 25, 10, 'Warm'),     # Warm Caramel
    (25, 20, 5, 'Warm'),      # Warm Amber

    # Deep warm tones
    (180, 130, 105, 'Warm'),  # Deep Warm 1
    (165, 120, 95, 'Warm'),   # Deep Warm 2
    (150, 110, 85, 'Warm'),   # Deep Warm 3
    (135, 100, 75, 'Warm'),   # Deep Warm 4
    (120, 90, 65, 'Warm'),    # Deep Warm 5
    (105, 80, 55, 'Warm'),    # Deep Warm 6
    (90, 70, 45, 'Warm'),     # Deep Warm 7
    (75, 60, 35, 'Warm'),     # Deep Warm 8
    (60, 50, 30, 'Warm'),     # Deep Warm 9
    (45, 40, 25, 'Warm'),     # Deep Warm 10

    # Additional variations
    (255, 220, 190, 'Warm'),  # Light Peach
    (240, 210, 175, 'Warm'),  # Light Golden
    (225, 200, 160, 'Warm'),  # Soft Caramel
    (210, 180, 140, 'Warm'),  # Warm Clay
    (190, 160, 120, 'Warm'),  # Golden Beige
    (175, 140, 100, 'Warm'),  # Tan Brown
    (160, 120, 85, 'Warm'),   # Honey Brown
    (145, 105, 70, 'Warm'),   # Warm Ochre
    (130, 90, 55, 'Warm'),    # Rustic Brown
    (115, 75, 45, 'Warm'),    # Medium Cocoa
    (100, 60, 40, 'Warm'),    # Earthy Brown
    (85, 50, 30, 'Warm'),     # Deep Chestnut

    # Additional Asian warm tones
    (255, 240, 220, 'Warm'),   # Warm Ivory
    (240, 215, 195, 'Warm'),   # Light Warm Beige
    (225, 200, 180, 'Warm'),   # Medium Warm Beige
    (210, 185, 165, 'Warm'),   # Medium-Dark Warm Beige
    (195, 170, 150, 'Warm'),   # Warm Tan
    (180, 155, 135, 'Warm'),   # Warm Almond
    (165, 140, 120, 'Warm'),   # Warm Caramel
    (150, 125, 105, 'Warm'),   # Warm Honey
    (135, 110, 90, 'Warm'),    # Warm Toffee
    (120, 95, 75, 'Warm'),     # Warm Bronze
    (105, 80, 60, 'Warm'),     # Warm Cocoa
    (90, 65, 50, 'Warm'),      # Warm Mahogany



    (255, 224, 189, 'Warm'),  # Light warm beige
    (240, 190, 140, 'Warm'),  # Warm peach
    (225, 170, 120, 'Warm'),  # Warm tan
    (210, 150, 100, 'Warm'),  # Warm brown
    (195, 130, 80, 'Warm'),   # Deep warm orange
    (180, 110, 60, 'Warm'),   # Warm caramel
    (165, 90, 50, 'Warm'),    # Rich warm bronze
    (150, 70, 40, 'Warm'),    # Warm mahogany
    (135, 50, 30, 'Warm'),    # Warm auburn
    (120, 40, 20, 'Warm'),     # Deep warm red

    (255, 220, 180, 'Warm'),  # Light Warm Beige
    (245, 200, 150, 'Warm'),  # Warm Sand
    (235, 190, 140, 'Warm'),  # Warm Honey
    (225, 180, 130, 'Warm'),  # Warm Caramel
    (215, 170, 120, 'Warm'),  # Warm Toffee
    (205, 160, 110, 'Warm'),  # Warm Almond
    (195, 150, 100, 'Warm'),  # Warm Mocha
    (185, 140, 90, 'Warm'),   # Warm Cocoa
    (175, 130, 80, 'Warm'),   # Warm Bronze
    (165, 120, 70, 'Warm'),   # Warm Chestnut
    (155, 110, 60, 'Warm'),   # Warm Copper
    (145, 100, 50, 'Warm'),   # Warm Maple



    # Cool undertones
    (240, 210, 210, 'Cool'),
    (210, 185, 170, 'Cool'),
    (190, 160, 150, 'Cool'),
    (150, 100, 90, 'Cool'),
    (110, 75, 70, 'Cool'),
    (245, 225, 230, 'Cool'),
    (255, 240, 250, 'Cool'),
    (230, 200, 210, 'Cool'),
    (210, 180, 185, 'Cool'),
    (180, 150, 155, 'Cool'),
    (140, 105, 110, 'Cool'),
    (100, 70, 75, 'Cool'),

    (255, 230, 235, 'Cool'),  # Very light cool
    (235, 200, 215, 'Cool'),  # Light-medium cool
    (215, 175, 190, 'Cool'),  # Medium cool
    (185, 145, 160, 'Cool'),  # Medium-dark cool
    (160, 120, 135, 'Cool'),  # Dark cool
    (130, 90, 100, 'Cool'),   # Very dark cool
    (255, 225, 245, 'Cool'),  # Lightest cool
    (240, 215, 225, 'Cool'),  # Medium cool
    (220, 195, 205, 'Cool'),  # Medium-dark cool
    (200, 170, 180, 'Cool'),  # Dark cool
    (170, 140, 150, 'Cool'),  # Darker cool
    (145, 110, 120, 'Cool'),  # Very dark cool

    (255, 240, 250, 'Cool'),  # Light cool
    (240, 215, 225, 'Cool'),  # Light-medium cool
    (225, 195, 205, 'Cool'),  # Medium cool
    (210, 170, 180, 'Cool'),  # Medium-dark cool
    (190, 145, 155, 'Cool'),  # Dark cool
    (170, 125, 135, 'Cool'),  # Darker cool
    (145, 100, 110, 'Cool'),  # Very dark cool
    (255, 225, 240, 'Cool'),  # Lightest cool

    (250, 230, 240, 'Cool'),  # Light pastel pink
    (235, 210, 230, 'Cool'),  # Light lavender
    (220, 190, 225, 'Cool'),  # Light mauve
    (205, 175, 220, 'Cool'),  # Medium lavender
    (190, 160, 210, 'Cool'),  # Medium periwinkle
    (175, 145, 200, 'Cool'),  # Medium violet
    (160, 130, 190, 'Cool'),  # Medium purple
    (145, 115, 180, 'Cool'),  # Medium plum
    (130, 100, 170, 'Cool'),  # Dark lavender
    (115, 85, 160, 'Cool'),   # Dark violet
    (100, 70, 150, 'Cool'),   # Dark purple
    (85, 55, 140, 'Cool'),    # Deep purple
    (70, 40, 130, 'Cool'),    # Deep violet
    (55, 30, 120, 'Cool'),    # Very deep purple
    (40, 25, 110, 'Cool'),    # Very deep violet
    (30, 20, 100, 'Cool'),    # Extremely deep purple
    (25, 15, 90, 'Cool'),     # Extremely deep violet
    (20, 10, 80, 'Cool'),     # Almost black violet
    (15, 5, 70, 'Cool'),      # Nearly black purple
    (10, 0, 60, 'Cool'),      # Blackish violet

    # Deep cool tones
    (130, 100, 90, 'Cool'),   # Deep Cool 1
    (115, 85, 75, 'Cool'),    # Deep Cool 2
    (100, 70, 60, 'Cool'),    # Deep Cool 3
    (85, 55, 45, 'Cool'),     # Deep Cool 4
    (70, 40, 30, 'Cool'),     # Deep Cool 5
    (55, 30, 20, 'Cool'),     # Deep Cool 6
    (40, 20, 10, 'Cool'),     # Deep Cool 7
    (30, 15, 10, 'Cool'),     # Deep Cool 8
    (20, 10, 5, 'Cool'),      # Deep Cool 9
    (10, 5, 0, 'Cool'),        # Deep Cool 10

    # Additional Asian cool tones
    (250, 235, 240, 'Cool'),    # Light Pastel Rose
    (235, 220, 230, 'Cool'),    # Light Lilac
    (220, 205, 225, 'Cool'),    # Medium Light Lavender
    (205, 190, 215, 'Cool'),    # Medium Lavender
    (190, 175, 205, 'Cool'),    # Medium Periwinkle
    (175, 160, 195, 'Cool'),    # Medium Violet
    (160, 145, 185, 'Cool'),    # Medium Purple
    (145, 130, 175, 'Cool'),    # Medium Plum
    (130, 115, 165, 'Cool'),    # Dark Lavender
    (115, 100, 155, 'Cool'),    # Dark Violet
    (100, 85, 145, 'Cool'),     # Dark Purple
    (85, 70, 135, 'Cool'),      # Deep Purple
    (70, 55, 125, 'Cool'),      # Deep Violet
    (55, 40, 115, 'Cool'),      # Very Deep Purple
    (40, 30, 105, 'Cool'),      # Very Deep Violet
    (30, 20, 95, 'Cool'),       # Extremely Deep Purple
    (25, 15, 85, 'Cool'),       # Extremely Deep Violet
    (20, 10, 75, 'Cool'),       # Almost Black Violet
    (15, 5, 65, 'Cool'),        # Nearly Black Purple
    (10, 0, 55, 'Cool'),        # Blackish Violet




    # Neutral undertones
    (245, 215, 200, 'Neutral'),
    (225, 190, 160, 'Neutral'),
    (200, 160, 130, 'Neutral'),
    (160, 110, 80, 'Neutral'),
    (120, 80, 60, 'Neutral'),
    (240, 220, 210, 'Neutral'),
    (230, 215, 200, 'Neutral'),
    (215, 190, 175, 'Neutral'),
    (195, 170, 155, 'Neutral'),
    (165, 130, 110, 'Neutral'),
    (135, 95, 80, 'Neutral'),
    (95, 60, 50, 'Neutral'),

    (255, 240, 230, 'Neutral'),  # Very light neutral
    (235, 210, 185, 'Neutral'),  # Light-medium neutral
    (210, 180, 155, 'Neutral'),  # Medium neutral
    (180, 140, 120, 'Neutral'),  # Medium-dark neutral
    (150, 110, 90, 'Neutral'),   # Dark neutral
    (130, 95, 75, 'Neutral'),    # Very dark neutral
    (250, 225, 215, 'Neutral'),  # Lightest neutral
    (230, 200, 180, 'Neutral'),  # Medium neutral
    (200, 170, 150, 'Neutral'),  # Medium-dark neutral
    (180, 150, 130, 'Neutral'),  # Dark neutral
    (155, 125, 105, 'Neutral'),  # Darker neutral
    (125, 95, 75, 'Neutral'),    # Very dark neutral

    (250, 220, 200, 'Neutral'),  # Very light neutral
    (235, 205, 180, 'Neutral'),  # Light-medium neutral
    (220, 190, 165, 'Neutral'),  # Medium neutral
    (200, 170, 150, 'Neutral'),  # Medium-dark neutral
    (180, 150, 130, 'Neutral'),  # Dark neutral
    (160, 130, 105, 'Neutral'),  # Very dark neutral
    (255, 235, 215, 'Neutral'),  # Lightest neutral
    (240, 215, 195, 'Neutral'),  # Medium neutral
    (225, 200, 175, 'Neutral'),  # Medium-dark neutral
    (210, 185, 155, 'Neutral'),  # Dark neutral
    (195, 165, 135, 'Neutral'),  # Darker neutral
    (180, 145, 115, 'Neutral'),  # Very dark neutral

    (240, 230, 220, 'Neutral'),  # Very light neutral
    (225, 210, 195, 'Neutral'),  # Light-medium neutral
    (210, 195, 180, 'Neutral'),  # Medium neutral
    (195, 180, 165, 'Neutral'),  # Medium-dark neutral
    (180, 165, 150, 'Neutral'),  # Dark neutral
    (165, 150, 135, 'Neutral'),  # Very dark neutral
    (150, 135, 120, 'Neutral'),  # Lightest neutral
    (135, 120, 105, 'Neutral'),  # Medium neutral
    (120, 105, 90, 'Neutral'),   # Medium-dark neutral
    (105, 90, 75, 'Neutral'),    # Dark neutral
    (90, 75, 60, 'Neutral'),     # Darker neutral
    (75, 60, 50, 'Neutral'),     # Very dark neutral
    (60, 50, 40, 'Neutral'),     # Deep neutral
    (50, 40, 30, 'Neutral'),     # Very deep neutral
    (40, 30, 20, 'Neutral'),     # Almost black neutral
    (30, 20, 15, 'Neutral'),     # Nearly black neutral
    (20, 15, 10, 'Neutral'),     # Extremely dark neutral
    (15, 10, 5, 'Neutral'),      # Extremely dark neutral
    (10, 5, 0, 'Neutral'),       # Almost black
    (5, 0, 0, 'Neutral'),        # Blackish neutral

    # Rich neutral tones
    (180, 140, 115, 'Neutral'),  # Rich Neutral 1
    (165, 125, 100, 'Neutral'),  # Rich Neutral 2
    (150, 110, 85, 'Neutral'),   # Rich Neutral 3
    (135, 100, 75, 'Neutral'),   # Rich Neutral 4
    (120, 90, 65, 'Neutral'),    # Rich Neutral 5
    (105, 80, 55, 'Neutral'),    # Rich Neutral 6
    (90, 70, 45, 'Neutral'),     # Rich Neutral 7
    (75, 60, 35, 'Neutral'),     # Rich Neutral 8
    (60, 50, 30, 'Neutral'),     # Rich Neutral 9
    (45, 40, 25, 'Neutral'),      # Rich Neutral 10

    # Additional Asian neutral tones
    (245, 230, 215, 'Neutral'),  # Light Neutral Beige
    (230, 210, 200, 'Neutral'),  # Medium Light Neutral
    (215, 195, 180, 'Neutral'),  # Medium Neutral
    (200, 180, 165, 'Neutral'),  # Medium-Dark Neutral
    (185, 165, 150, 'Neutral'),  # Neutral Tan
    (170, 150, 135, 'Neutral'),  # Neutral Almond
    (155, 140, 125, 'Neutral'),  # Neutral Caramel
    (140, 125, 110, 'Neutral'),  # Neutral Honey
    (125, 110, 95, 'Neutral'),   # Neutral Toffee
    (110, 95, 80, 'Neutral'),    # Neutral Bronze
    (95, 80, 65, 'Neutral'),     # Neutral Cocoa
    (80, 65, 50, 'Neutral'),     # Neutral Mahogany

    (245, 215, 200, 'Neutral'),
    (225, 190, 160, 'Neutral'),
    (200, 160, 130, 'Neutral'),
    (160, 110, 80, 'Neutral'),
    (120, 80, 60, 'Neutral'),
    (240, 220, 210, 'Neutral'),
    (230, 215, 200, 'Neutral'),
    (215, 190, 175, 'Neutral'),
    (195, 170, 155, 'Neutral'),
    (165, 130, 110, 'Neutral'),
    (135, 95, 80, 'Neutral'),
    (95, 60, 50, 'Neutral'),
    (255, 240, 230, 'Neutral'),
    (235, 210, 185, 'Neutral'),
    (210, 180, 155, 'Neutral'),
    (180, 140, 120, 'Neutral'),
    (150, 110, 90, 'Neutral'),
    (130, 95, 75, 'Neutral'),
    (250, 225, 215, 'Neutral'),
    (230, 200, 180, 'Neutral'),
    (200, 170, 150, 'Neutral'),
    (180, 150, 130, 'Neutral'),
    (155, 125, 105, 'Neutral'),
    (125, 95, 75, 'Neutral'),
    (250, 220, 200, 'Neutral'),
    (235, 205, 180, 'Neutral'),
    (220, 190, 165, 'Neutral'),
    (200, 170, 150, 'Neutral'),
    (180, 150, 130, 'Neutral'),
    (255, 235, 215, 'Neutral'),
    (240, 215, 195, 'Neutral'),
    (225, 200, 175, 'Neutral'),
    (210, 185, 155, 'Neutral'),
    (195, 165, 135, 'Neutral'),
    (180, 145, 115, 'Neutral'),
    (240, 230, 220, 'Neutral'),
    (225, 210, 195, 'Neutral'),
    (210, 195, 180, 'Neutral'),
    (195, 180, 165, 'Neutral'),
    (180, 165, 150, 'Neutral'),
    (165, 150, 135, 'Neutral'),
    (150, 135, 120, 'Neutral'),
    (135, 120, 105, 'Neutral'),
    (120, 105, 90, 'Neutral'),
    (105, 90, 75, 'Neutral'),
    (90, 75, 60, 'Neutral'),
    (75, 60, 50, 'Neutral'),
    (60, 50, 40, 'Neutral'),
    (50, 40, 30, 'Neutral'),
    (40, 30, 20, 'Neutral'),
    (30, 20, 15, 'Neutral'),
    (20, 15, 10, 'Neutral'),
    (15, 10, 5, 'Neutral'),
    (10, 5, 0, 'Neutral'),
    (5, 0, 0, 'Neutral'),
    (180, 140, 115, 'Neutral'),
    (165, 125, 100, 'Neutral'),
    (150, 110, 85, 'Neutral'),
    (135, 100, 75, 'Neutral'),
    (120, 90, 65, 'Neutral'),
    (105, 80, 55, 'Neutral'),
    (90, 70, 45, 'Neutral'),
    (75, 60, 35, 'Neutral'),
    (60, 50, 30, 'Neutral'),
    (45, 40, 25, 'Neutral'),
    (245, 230, 215, 'Neutral'),
    (230, 210, 200, 'Neutral'),
    (215, 195, 180, 'Neutral'),
    (200, 180, 165, 'Neutral'),
    (185, 165, 150, 'Neutral'),
    (170, 150, 135, 'Neutral'),
    (155, 140, 125, 'Neutral'),
    (140, 125, 110, 'Neutral'),
    (125, 110, 95, 'Neutral'),
    (110, 95, 80, 'Neutral'),
    (95, 80, 65, 'Neutral'),
    (80, 65, 50, 'Neutral'),

    (230, 220, 205, 'Neutral'),  # Soft neutral beige
    (215, 200, 185, 'Neutral'),  # Light neutral brown
    (200, 185, 170, 'Neutral'),  # Neutral tan
    (185, 170, 155, 'Neutral'),  # Medium neutral brown
    (170, 155, 140, 'Neutral'),  # Dark neutral beige
    (155, 140, 125, 'Neutral'),  # Rich neutral brown
    (140, 125, 110, 'Neutral'),  # Deep neutral beige
    (125, 110, 95, 'Neutral'),   # Very deep neutral brown
    (110, 95, 80, 'Neutral'),    # Dark brown
    (95, 80, 65, 'Neutral'),      # Deep brown

    (255, 250, 240, 'Neutral'),  # Very Light Neutral Beige
    (245, 235, 220, 'Neutral'),  # Light Neutral Cream
    (235, 225, 210, 'Neutral'),  # Light Neutral Sand
    (225, 215, 200, 'Neutral'),  # Light Neutral Tan
    (215, 205, 190, 'Neutral'),  # Medium-Light Neutral
    (205, 195, 180, 'Neutral'),  # Medium Neutral Beige
    (195, 185, 170, 'Neutral'),  # Medium Neutral Tan
    (185, 175, 160, 'Neutral'),  # Medium Neutral Buff
    (175, 165, 150, 'Neutral'),  # Medium Neutral Almond
    (165, 155, 140, 'Neutral'),  # Medium Neutral Caramel
    (155, 145, 130, 'Neutral'),  # Medium Neutral Cocoa
    (145, 135, 120, 'Neutral'),  # Medium Neutral Chestnut
    (135, 125, 110, 'Neutral'),  # Medium-Dark Neutral
    (125, 115, 100, 'Neutral'),  # Dark Neutral Tan
    (115, 105, 90, 'Neutral'),   # Dark Neutral Mocha
    (105, 95, 80, 'Neutral'),    # Dark Neutral Coffee
    (95, 85, 70, 'Neutral'),     # Dark Neutral Cocoa
    (85, 75, 60, 'Neutral'),     # Very Dark Neutral
    (75, 65, 50, 'Neutral'),     # Deep Neutral
    (65, 55, 40, 'Neutral'),     # Very Deep Neutral
    (55, 45, 30, 'Neutral'),     # Almost Black Neutral
    (45, 35, 20, 'Neutral'),     # Nearly Black Neutral
    (35, 25, 15, 'Neutral'),     # Extremely Dark Neutral
    (25, 15, 10, 'Neutral'),     # Extremely Dark Neutral


    ]

        def remove_duplicates(color_list):
            """Removes duplicate color tuples from a list."""
            unique_colors = list(set(color_list))
            return unique_colors

        # Process the dataset
        dataset = remove_duplicates(data)
        X = np.array([d[:3] for d in dataset])  # RGB values
        y = np.array([d[3] for d in dataset])   # Season labels

        # Normalize the RGB values
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_index, test_index in sss.split(X_normalized, y):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Train the KNN model
        knn_rgb = KNeighborsClassifier(n_neighbors=4, p=2, weights='distance')
        knn_rgb.fit(X_train, y_train)

        # Store the trained model and scaler
        predict_skin_tone_classification.model = knn_rgb
        predict_skin_tone_classification.scaler = scaler

    # Normalize the new average skin color
    average_skin_color_normalized = predict_skin_tone_classification.scaler.transform([average_skin_color])

    # Predict and return the classification
    predicted_classification = predict_skin_tone_classification.model.predict(average_skin_color_normalized)
    return predicted_classification[0]

predicted_classification = predict_skin_tone_classification(average_skin_color)
print(f"The predicted skin tone is {predicted_classification}")




def predict_season(average_eye_color, average_hair_color, average_lip_color, average_brows_color):
    """
    Predicts the season label based on the average color values of eye, hair, lip, and brows.

    Parameters:
    average_eye_color (list or tuple): RGB values for eye color.
    average_hair_color (list or tuple): RGB values for hair color.
    average_lip_color (list or tuple): RGB values for lip color.
    average_brows_color (list or tuple): RGB values for brow color.

    Returns:
    str: The predicted season label.
    """
    # Static dataset and labels
    data = np.array([
             ###-----Winter-------###




    [60, 80, 120, 20, 20, 20, 180, 30, 50, 30, 30, 30],
    [70, 90, 130, 10, 10, 10, 150, 20, 40, 20, 20, 20],
    [50, 70, 110, 30, 30, 30, 200, 40, 60, 40, 40, 40],
    [80, 100, 140, 20, 20, 30, 170, 50, 70, 25, 25, 25],
    [90, 110, 150, 15, 15, 25, 160, 45, 65, 35, 35, 35],

        # Caucasian - Cool, high contrast
    [70, 80, 130, 30, 30, 30, 190, 40, 50, 30, 30, 30],
    [80, 90, 140, 25, 25, 25, 180, 30, 40, 35, 35, 35],

    # African - Cool, dark contrast
    [50, 60, 80, 10, 10, 10, 130, 20, 30, 20, 20, 20],
    [60, 70, 90, 15, 15, 15, 120, 30, 40, 25, 25, 25],

    # Asian - Cool, high contrast
    [50, 60, 100, 20, 20, 20, 160, 30, 45, 20, 20, 20],
    [55, 65, 110, 25, 25, 25, 170, 35, 50, 25, 25, 25],

    # Hispanic/Latino - Cool, high contrast
    [65, 75, 110, 25, 25, 25, 150, 35, 45, 25, 25, 25],
    [60, 70, 105, 30, 30, 30, 160, 40, 50, 30, 30, 30],

    # Middle Eastern - Cool, high contrast
    [70, 80, 90, 20, 20, 30, 140, 30, 40, 30, 30, 30],
    [80, 90, 100, 25, 25, 35, 150, 35, 45, 35, 35, 35],



                ###-----Summer-------###


    # Caucasian - Cool, soft
    [160, 170, 190, 140, 140, 150, 220, 150, 160, 100, 90, 80],
    [150, 160, 180, 130, 130, 140, 210, 140, 150, 95, 85, 75],

    # African - Soft contrast
    [120, 130, 140, 70, 70, 80, 180, 80, 90, 60, 50, 50],
    [130, 140, 150, 80, 80, 90, 190, 90, 100, 65, 55, 55],

    # Asian - Soft, cool
    [150, 160, 170, 90, 90, 100, 200, 100, 110, 75, 65, 60],
    [140, 150, 160, 85, 85, 95, 190, 95, 105, 70, 60, 55],

    # Hispanic/Latino - Cool, soft
    [140, 150, 160, 100, 100, 110, 200, 110, 120, 80, 70, 65],
    [135, 145, 155, 95, 95, 105, 190, 105, 115, 75, 65, 60],

        # Middle Eastern - Soft, muted
    [130, 140, 150, 100, 100, 110, 170, 120, 130, 80, 70, 65],
    [140, 150, 160, 105, 105, 115, 180, 125, 135, 85, 75, 70],
    [160, 180, 200, 150, 150, 140, 220, 140, 160, 100, 90, 80],
    [150, 170, 190, 140, 140, 130, 210, 130, 150, 110, 100, 90],
    [155, 175, 195, 130, 130, 120, 200, 120, 140, 105, 95, 85],
    [140, 160, 180, 160, 160, 150, 230, 150, 170, 115, 105, 95],
    [165, 185, 205, 135, 135, 125, 190, 110, 130, 95, 85, 75],



                        ###-----Autumn-------###


    [120, 80, 40, 100, 50, 20, 180, 70, 50, 60, 40, 20],
    [130, 90, 50, 110, 60, 30, 190, 80, 60, 70, 50, 30],
    [110, 70, 30, 90, 40, 10, 170, 60, 40, 50, 30, 10],
    [140, 100, 60, 120, 70, 40, 200, 90, 70, 80, 60, 40],
    [115, 75, 35, 95, 45, 15, 185, 65, 45, 55, 35, 15],

    # Caucasian - Warm, muted
    [130, 90, 50, 110, 80, 40, 190, 90, 60, 70, 60, 30],
    [120, 80, 40, 100, 70, 30, 180, 80, 50, 65, 55, 25],

    # African - Warm, earthy
    [90, 60, 40, 30, 20, 10, 150, 80, 70, 35, 25, 15],
    [100, 70, 50, 40, 30, 20, 160, 85, 75, 40, 30, 20],

    # Asian - Warm, muted
    [110, 70, 40, 60, 40, 30, 170, 70, 60, 55, 45, 35],
    [120, 80, 50, 70, 50, 40, 180, 80, 70, 60, 50, 40],

    # Hispanic/Latino - Warm, earthy
    [130, 100, 70, 80, 50, 40, 190, 90, 80, 70, 60, 50],
    [140, 110, 80, 90, 60, 50, 200, 100, 90, 75, 65, 55],

    # Middle Eastern - Warm, rich
    [115, 85, 60, 80, 60, 40, 180, 80, 60, 60, 50, 35],
    [125, 95, 70, 90, 70, 50, 190, 90, 70, 65, 55, 40],




                 ###---- Spring-------###


    [180, 200, 80, 220, 180, 120, 240, 130, 100, 180, 140, 100],
    [190, 210, 90, 230, 190, 130, 250, 140, 110, 190, 150, 110],
    [170, 190, 70, 210, 170, 110, 230, 120, 90, 170, 130, 90],
    [200, 220, 100, 240, 200, 140, 260, 150, 120, 200, 160, 120],
    [160, 180, 60, 200, 160, 100, 220, 110, 80, 160, 120, 80],

        # Caucasian - Warm, clear
    [180, 200, 80, 220, 190, 130, 240, 160, 100, 180, 150, 100],
    [170, 190, 70, 210, 180, 120, 230, 150, 90, 170, 140, 90],

    # African - Warm, vibrant
    [140, 90, 50, 40, 30, 20, 190, 120, 110, 60, 50, 40],
    [150, 100, 60, 50, 40, 30, 200, 130, 120, 70, 60, 50],

    # Asian - Warm, clear
    [160, 120, 80, 70, 50, 40, 220, 150, 100, 80, 60, 50],
    [170, 130, 90, 80, 60, 50, 230, 160, 110, 90, 70, 60],


    # Hispanic/Latino - Warm, vibrant
    [190, 150, 110, 140, 100, 80, 250, 180, 130, 110, 90, 80],
    [180, 140, 100, 130, 90, 70, 240, 170, 120, 100, 80, 70],

    # Middle Eastern - Warm, clear
    [200, 160, 110, 120, 80, 60, 240, 160, 110, 110, 90, 70],
    [190, 150, 100, 110, 70, 50, 230, 150, 100, 100, 80, 60]
])
    num_points_per_season = 15
    labels = np.array([
        0] * num_points_per_season +  # Winter
        [1] * num_points_per_season +  # Summer
        [2] * num_points_per_season +  # Autumn
        [3] * num_points_per_season    # Spring
    )

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # Adjust k as needed
    knn.fit(X_train, y_train)

    # Flatten the average color values into a feature vector
    average_features = np.array([
        average_eye_color[0], average_eye_color[1], average_eye_color[2],  # Eye color
        average_hair_color[0], average_hair_color[1], average_hair_color[2],  # Hair color
        average_lip_color[0], average_lip_color[1], average_lip_color[2],  # Lip color
        average_brows_color[0], average_brows_color[1], average_brows_color[2]   # Brow color
    ])

    # Normalize the average features
    average_features_normalized = scaler.transform([average_features])

    # Define label mapping
    label_mapping = {
        0: 'Winter',
        1: 'Summer',
        2: 'Autumn',
        3: 'Spring'
    }

    # Predict the season
    prediction = knn.predict(average_features_normalized)

    # Map the prediction to a human-readable label
    predicted_season = label_mapping[prediction[0]]
    return predicted_season



predicted_season = predict_season(average_eye_color, average_hair_color, average_lip_color, average_brows_color)
print("Predicted Season:", predicted_season)



### to be copied out


import matplotlib.pyplot as plt
import numpy as np

def color_palette_recommendation_and_visualization(predicted_classification, predicted_season):
    """
    Recommend a color palette and visualize it based on the predicted skin tone and season classification.

    Args:
    predicted_classification (list of str): List containing the predicted skin tone.
    predicted_season (str): The predicted season.

    Returns:
    None
    """

    warm_winter_palette = [
    (0, 191, 255),    # Deep Sky Blue
    (255, 20, 147),   # Deep Pink
    (255, 69, 0),     # Red-Orange
    (0, 255, 255),    # Cyan
    (135, 206, 250),  # Light Sky Blue
    (255, 105, 180),  # Hot Pink
    (0, 0, 255),      # Blue
    (255, 0, 0),      # Red
    (240, 248, 255),  # Alice Blue
    (173, 216, 230),  # Light Blue
    (255, 250, 250),  # Snow
    (224, 255, 255),  # Light Cyan
    (135, 206, 255),  # Sky Blue
    (255, 228, 225)   # Misty Rose
    ]

    neutral_winter_palette = [
    (0, 0, 128),      # Navy
    (0, 0, 255),      # Blue
    (25, 25, 112),    # Midnight Blue
    (75, 0, 130),     # Indigo
    (128, 0, 128),    # Purple
    (135, 206, 250),  # Light Sky Blue
    (70, 130, 180),   # Steel Blue
    (0, 255, 255),    # Cyan
    (0, 139, 139),    # Dark Cyan
    (255, 255, 255),  # White
    (192, 192, 192),  # Silver
    (139, 0, 255),    # Dark Violet
    (0, 206, 209),    # Dark Turquoise
    (0, 191, 255)     # Deep Sky Blue
    ]

    cool_winter_palette = [
    (48, 25, 52),     # Deep Plum
    (72, 61, 139),    # Dark Slate Blue
    (80, 0, 128),     # Dark Purple
    (47, 79, 79),     # Dark Slate Gray
    (64, 64, 64),     # Charcoal Gray
    (0, 0, 128),      # Navy
    (128, 0, 128),    # Purple
    (0, 139, 139),    # Dark Cyan
    (139, 69, 19),    # Saddle Brown
    (102, 51, 153),   # Rebecca Purple
    (105, 105, 105),  # Dim Gray
    (0, 0, 139),      # Dark Blue
    (139, 0, 0),      # Dark Red
    (46, 139, 87)     # Sea Green
    ]

    # Define the True Summer color palette
    warm_summer_palette = [
        # Muted Blues
        (100, 149, 237),  # Cornflower Blue (muted blue)
        (176, 224, 230),  # Powder Blue (pastel blue)

        # Muted Purples
        (147, 112, 219),  # Medium Purple
        (200, 162, 200),  # Lilac (pastel purple)

        # Muted Pinks
        (255, 192, 203),  # Light Pink (muted pink)
        (255, 105, 180),  # Hot Pink (lighter pink)

        # Neutrals
        (183, 183, 183),  # Pearl Grey
        (191, 191, 191),  # Dove Grey
        (255, 255, 240),  # Ivory

        # Air Force Blue
        (93, 138, 168),  # Air Force Blue

        # Light Reds
        (227, 11, 92),   # Raspberry
        (255, 127, 80),  # Coral

        # Pastels
        (46, 139, 87),   # Sea Green (pastel green)
        (255, 220, 185)  # Nude Pink (pastel pink)
    ]

    neutral_summer_palette = [
        # Light Summer Pastels (5 colors)
        (127, 255, 212),  # Aquamarine Blue
        (204, 204, 255),  # Light Purple (Wisteria or Lavender)
        (152, 255, 152),  # Mint Green
        (255, 240, 245),  # Lavender Blush (light pink)
        (240, 248, 255),  # Alice Blue (pale blue)

        # Cool Summer Colors (5 colors)
        (64, 224, 208),   # Tiffany Blue
        (144, 238, 144),  # Light Green
        (227, 11, 92),    # Raspberry Red
        (135, 206, 250),  # Light Sky Blue
        (100, 149, 237),  # Cornflower Blue

        # Additional colors to balance to 14 (4 colors)
        (176, 224, 230),  # Powder Blue (cool pastel blue)
        (153, 50, 204),   # Dark Orchid (brighter purple)
        (0, 191, 255),    # Deep Sky Blue (brighter blue)
        (60, 179, 113)    # Medium Sea Green (pastel green)
        ]


    cool_summer_palette = [
        # Cool and Pale Shades
        (211, 211, 211),  # Light Grey
        (152, 255, 152),  # Mint Green
        (0, 128, 0),      # Military Green
        (230, 230, 250),  # Lavender (pale purple)
        (192, 192, 192),  # Silver
        (128, 128, 128),  # Grey
        (255, 240, 245),  # Lavender Blush (pale pink)

        # Autumnal Shades
        (194, 178, 128),  # Sand
        (169, 169, 169),  # Warm Dove Grey
        (255, 228, 196),  # Bisque (warm beige)

        # Additional Colors
        (255, 215, 0),    # Gold
        (85, 107, 47),    # Dark Olive Green
        (0, 128, 128),    # Teal
        (102, 205, 170)   # Medium Aquamarine
    ]

    warm_autumn_palette = [
        # Warm Greens
        (34, 139, 34),    # Forest Green
        (75, 83, 32),     # Army Green
        (173, 223, 173),  # Moss Green
        (128, 128, 0),    # Olive Green
        (140, 255, 0),    # Kiwi Green

        # Golden Yellows
        (255, 215, 0),    # Gold
        (255, 204, 0),    # Golden Yellow
        (255, 193, 37),   # Sunflower Yellow

        # Terracotta and Burnt Oranges
        (204, 102, 0),    # Terracotta
        (255, 87, 34),    # Burnt Orange
        (255, 140, 0),    # Dark Orange

        # Deep Reds
        (139, 0, 0),      # Dark Red
        (255, 0, 0),      # Red
        (205, 92, 92),     # Indian Red

        # Oyster
        (217, 217, 217),  # Oyster (light grey)

        # Coffee Brown
        (111, 78, 55)     # Coffee Brown
    ]

    neutral_autumn_palette = [
        # Taupes and Soft Browns
        (139, 126, 102),  # Taupe
        (186, 143, 103),  # Light Coffee
        (188, 143, 143),  # Rosy Brown
        (210, 180, 140),  # Light Tan
        (169, 123, 92),   # Walnut Brown

        # Soft Purples
        (147, 112, 219),  # Medium Purple
        (138, 43, 226),   # Blue Violet
        (186, 85, 211),   # Medium Orchid
        (216, 191, 216),  # Thistle (light lavender)

        # Soft Greens and Teals
        (143, 188, 143),  # Pale Green
        (0, 128, 128),    # Teal
        (102, 205, 170),  # Medium Aquamarine
        (152, 251, 152),   # Pale Green

        # Accent Colors
        (255, 105, 180),  # Hot Pink (Accent)
        (255, 140, 0)     # Dark Orange (Accent)
    ]

    cool_autumn_palette = [
        # Deep Neutrals
        (0, 0, 128),      # Warm Navy
        (255, 255, 255),  # Soft White
        (105, 105, 105),  # Dim Grey
        (70, 70, 70),     # Dark Grey

        # Bold Reds
        (139, 0, 0),      # Dark Red
        (255, 0, 0),      # Red
        (255, 99, 71),    # Tomato

        # Deep Greens
        (0, 128, 0),      # Green
        (0, 100, 0),      # Dark Green
        (46, 139, 87),    # Sea Green

        # Teals
        (0, 128, 128),    # Teal
        (0, 139, 139)     # Dark Cyan
    ]

    warm_spring_palette = [
        # Warm Greens
        (152, 251, 152),  # Pale Green
        (0, 128, 0),      # Green
        (102, 204, 102),  # Medium Spring Green

        # Yellows
        (255, 255, 0),    # Yellow
        (255, 255, 224),  # Light Yellow
        (255, 215, 0),    # Gold

        # Orangey Reds
        (255, 69, 0),     # Red-Orange (or Tomato)
        (255, 99, 71),    # Tomato
        (255, 140, 0),    # Dark Orange

        # Peachy Pinks
        (255, 192, 203),  # Pink
        (255, 218, 185),  # Peach Puff
        (255, 160, 122),  # Light Salmon

        # Browns
        (139, 69, 19),    # Saddle Brown
        (205, 133, 63),    # Peru

        # Teal and Aquamarine
        (0, 128, 128),    # Teal
        (127, 255, 212)   # Aquamarine
    ]

    # Define the Neutral Spring color palette with exactly 14 colors
    neutral_spring_palette = [
        # Bright and Warm Yellows
        (255, 255, 0),    # Yellow
        (255, 255, 224),  # Light Yellow
        (255, 215, 0),    # Gold

        # Bright and Warm Greens
        (144, 238, 144),  # Light Green
        (0, 255, 0),      # Lime
        (0, 128, 0),      # Green

        # Bright and Warm Blues
        (173, 216, 230),  # Light Blue
        (135, 206, 250),  # Light Sky Blue
        (0, 0, 255),      # Blue

        # Bright and Warm Reds
        (255, 99, 71),    # Tomato
        (255, 69, 0),     # Red-Orange
        (255, 0, 0),      # Red

        # Neutrals
        (255, 240, 245),  # Lavender Blush
        (240, 248, 255),  # Alice Blue
        (255, 255, 255),   # White

        # Neutrals
        (255, 240, 245),  # Lavender Blush
        (240, 248, 255),  # Alice Blue
        (255, 228, 196),  # Biscotti
        (138, 43, 226)    # Violet Island Bird
    ]


    cool_spring_palette = [
        # Clear and Bright Blues
        (0, 191, 255),    # Deep Sky Blue
        (0, 0, 255),      # Blue
        (135, 206, 250),  # Light Sky Blue

        # Sharp Yellows
        (255, 255, 0),    # Yellow
        (255, 215, 0),    # Gold
        (255, 255, 224),  # Light Yellow

        # Almost-Neon Pinks and related
        (255, 99, 71),    # Tomato (Bright red-orange)
        (255, 0, 0),      # Red
        (255, 182, 193),  # Light Pink
        (255, 87, 34),    # Strawberry

        # Additional Colors
        (255, 20, 147),   # Deep Pink (Confetti)
        (152, 251, 152),  # Mint
        (192, 192, 192),  # Silver Marl
        (169, 169, 169),  # Dove Grey

        # Neutrals
        (240, 248, 255),  # Alice Blue
        (255, 250, 250)   # Snow
    ]

    # Define the palettes dictionary
    palettes = {
        ("warm", "spring"): warm_spring_palette,
        ("warm", "autumn"): warm_autumn_palette,
        ("warm", "summer"): warm_summer_palette,
        ("warm", "winter"): warm_winter_palette,
        ("neutral", "spring"): neutral_spring_palette,
        ("neutral", "autumn"): neutral_autumn_palette,
        ("neutral", "summer"): neutral_summer_palette,
        ("neutral", "winter"): neutral_winter_palette,
        ("cool", "spring"): cool_spring_palette,
        ("cool", "autumn"): cool_autumn_palette,
        ("cool", "summer"): cool_summer_palette,
        ("cool", "winter"): cool_winter_palette
    }

    # Extract skin tone and season from predictions
    skin_tone = predicted_classification.lower()
    season = predicted_season.lower()

    # Lookup the appropriate palette
    recommended_palette = palettes.get((skin_tone, season), [])

    # Visualize the color palette
    def visualize_color(palette, title):
        """
        Visualize the color palette.

        Args:
        palette (list of tuple): The color palette to visualize.
        title (str): The title for the visualization.
        """
        if not palette:
            print("No palette available for the given classification and season.")
            return

        # Convert palette to a format suitable for imshow
        palette = np.array(palette).reshape(1, -1, 3) / 255.0  # Normalize RGB values to [0, 1]

        plt.figure(figsize=(10, 2))
        plt.imshow(palette, aspect='auto')
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Create the title for visualization
    title = f"Recommended {skin_tone.capitalize()} {season.capitalize()} Palette"

    # Visualize the recommended palette
    visualize_color(recommended_palette, title)

color_palette_recommendation_and_visualization(predicted_classification, predicted_season)
