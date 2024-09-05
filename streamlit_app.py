import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import json
import torch
from torch import nn

from image_processing import predict
from colour_palette import extract_average_colors, predict_skin_tone_classification ,predict_season,color_palette_recommendation_and_visualization, visualize_color
import cv2
import os



# Set custom CSS for font style, centering, and text box styling
# Set custom CSS for professional design, gradient backgrounds, and refined font styles
import streamlit as st

# Add your CSS styles
st.markdown(
    """
    <style>
    /* General body and app background */
    .stApp {
        background: linear-gradient(135deg, #f0f0f0, #d9d9d9);  /* Soft gradient for the page background */
        font-family: 'Arial, sans-serif';  /* Clean sans-serif font for the entire page */
        color: #333333;  /* Standardized dark grey for text */
    }

    /* Centered title and content */
    .serif-font {
        font-family: 'Georgia, serif';  /* Professional serif font for headings */
        text-align: center;
        color: #2c3e50;  /* Deep blue-grey color for headings */
    }

    /* Refined text box styling */
    .centered-box {
        background: linear-gradient(135deg, #ffffff, #e0e0e0);  /* Subtle gradient for content boxes */
        color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);  /* Soft shadow for depth */
        border: 1px solid #dcdcdc;  /* Light border for structure */
    }

    /* Palette boxes with refined design */
    .palette-box {
        background: linear-gradient(135deg, #ffffff, #f7f7f7);  /* Light gradient for subtle contrast */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #cccccc;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);  /* Softer shadow for light depth */
    }

    /* Heading style inside palette boxes */
    h2 {
        font-family: 'Georgia, serif';  /* Sophisticated serif font for palette titles */
        color: #2c3e50;  /* Darker, refined color for titles */
    }

    /* Soft link styling */
    a {
        color: #3498db;  /* Soft blue for clickable links */
        text-decoration: none;
    }
    a:hover {
        color: #2980b9;  /* Slightly darker on hover for better interaction feedback */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a description box
st.markdown(
    """
    <div class="centered-box">
        <h2>PCA</h2>
        <p>Using face segmentation. We create an identical mask where we can process hex values!</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Define the relative path to the images directory from streamlit_app.py
images_path = os.path.join("image_website")

# Create two columns for the images
col1, col2 = st.columns(2)

# Define the image paths
image1_path = os.path.join(images_path, "default_img.png")
image2_path = os.path.join(images_path, "default_mask.png")

# Add images to the columns
with col1:
    st.image(image1_path, use_column_width=True)

with col2:
    st.image(image2_path, use_column_width=True)


# Define color palettes for each season
summer_palette = [
    "rgb(207, 183, 207)",  # Dusty Rose
    "rgb(178, 223, 216)",  # Soft Teal
    "rgb(170, 204, 255)"   # Powder Blue
]

winter_palette = [
    "rgb(173, 216, 230)",  # Icy Blue
    "rgb(0, 0, 0)",        # Deep Black
    "rgb(255, 0, 0)"       # Vivid Red
]

fall_palette = [
    "rgb(255, 140, 0)",    # Burnt Orange
    "rgb(139, 69, 19)",    # Dark Chocolate
    "rgb(107, 142, 35)"    # Olive Green
]

spring_palette = [
    "rgb(255, 218, 185)",  # Peach
    "rgb(255, 255, 224)",  # Light Yellow
    "rgb(255, 165, 0)"     # Coral
]

# Generate CSS for gradient backgrounds for each palette
palette_styles = {
    'summer': f"linear-gradient(to right, {', '.join(summer_palette)})",
    'winter': f"linear-gradient(to right, {', '.join(winter_palette)})",
    'fall': f"linear-gradient(to right, {', '.join(fall_palette)})",
    'spring': f"linear-gradient(to right, {', '.join(spring_palette)})",
}

st.markdown(
    """
    <style>
    .serif-font {
        font-family: 'serif';
        text-align: center;
        color: #333;
    }
    .centered-box {
        background-color: #ffffff;
        color: #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .palette-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .summer-box { background: """ + palette_styles['summer'] + """; }
    .winter-box { background: """ + palette_styles['winter'] + """; }
    .fall-box { background: """ + palette_styles['fall'] + """; }
    .spring-box { background: """ + palette_styles['spring'] + """; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    '<h1 class="serif-font">Professional Color Recommendation</h1>',
    unsafe_allow_html=True
)

# Description
st.markdown(
    '<p class="serif-font">Personalize your natural appearance with our advanced color analysis tool.</p>',
    unsafe_allow_html=True
)

# Detailed description
st.markdown(
    '<div class="centered-box">This advanced tool utilizes facial segmentation to analyze your features, including your eyes, hair, eyebrows, lips, and skin tone. Each of these elements is evaluated to find the most complementary color palette for you.Our color analysis is based on seasonal theory, categorizing palettes by Summer, Winter, Fall, and Spring and further into warm, neutral, cool.</div>',
    unsafe_allow_html=True
)

# Summer Palette box
st.markdown(
    """
    <div class="palette-box summer-box">
    <h2>Summer Palette</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Winter Palette box
st.markdown(
    """
    <div class="palette-box winter-box">
    <h2>Winter Palette</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Fall (Autumn) Palette box
st.markdown(
    """
    <div class="palette-box fall-box">
    <h2>Fall (Autumn) Palette</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Spring Palette box
st.markdown(
    """
    <div class="palette-box spring-box">
    <h2>Spring Palette</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Face Parsing with Segformer")

# Image upload
img = st.camera_input("Take a picture")

if img is not None:
# Load image
        image = Image.open(img)


        #color extraction
        colored_mask,skin,average_skin_color, average_brows_color, average_hair_color, average_lip_color, average_eye_color = extract_average_colors(image)


        # Optionally, you can use st.columns to display the images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        with col2:
            st.image(colored_mask, caption='Colored Mask', use_column_width=True)

        #predict warm neutral or cool
        predicted_classification = predict_skin_tone_classification(extract_average_colors(image)[2])

        #predict which season
        predicted_season = predict_season(average_eye_color,average_hair_color, average_lip_color, average_brows_color)

        if img is not None:



            # Display the segmentation mask (optional: apply a color map)
            # Define the colors and their labels
            colors = [
                (average_skin_color, "Skin"),
                (average_brows_color, "Brows"),
                (average_hair_color, "Hair"),
                (average_eye_color, "Eyes"),
                (average_lip_color, "Lips")
            ]

            # Generate dynamic CSS for color boxes
            color_box_styles = ""
            for i in range(len(colors)):
                color = colors[i][0]
                color_box_styles += f"""
                .color-box-{i} {{
                    background-color: rgb({color[0]}, {color[1]}, {color[2]});
                    width: 100px;
                    height: 100px;
                    border-radius: 10px;
                    display: inline-block;
                    margin: 5px;
                    vertical-align: middle;
                }}
                .color-label-{i} {{
                    color: rgb({color[0]}, {color[1]}, {color[2]});  /* Use the same color for text */
                    text-align: center;
                    margin-top: 5px;
                    font-family: 'serif';  /* Ensure consistent font styling */
                    font-size: 14px;  /* Adjust font size as needed */
                }}
                """

            # Apply the CSS styles
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-color: #f0f0f0;  /* Light background for better contrast */
                    color: #333;  /* Dark text color for readability */
                }}
                {color_box_styles}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the color boxes with dynamic styling
            columns = st.columns(len(colors))
            for i, (color, title) in enumerate(colors):
                with columns[i]:
                    # Use HTML and CSS to display the color box and label
                    st.markdown(f'<div class="color-box-{i}"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="color-label-{i}">{title}</div>', unsafe_allow_html=True)

            #output palette
            palette = color_palette_recommendation_and_visualization(predicted_classification,predicted_season)

            columns = st.columns(len(palette))

            # Assuming the color palette is extracted after image analysis and stored in 'palette'

            # Display the colors dynamically in styled boxes
            columns = st.columns(len(palette))
            for i, color in enumerate(palette):
                with columns[i]:
                    st.markdown(f'''
                    <div style="background-color: rgb({color[0]}, {color[1]}, {color[2]});
                                width: 100px; height: 100px; border-radius: 10px;
                                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);">
                    </div>''',
                    unsafe_allow_html=True)

            # Dynamic CSS styling based on the extracted color palette
            background_color = f'rgb({palette[0][0]}, {palette[0][1]}, {palette[0][2]})'  # First color for the background
            text_color = f'rgb({palette[1][0]}, {palette[1][1]}, {palette[1][2]})'       # Second color for text
            box_background_color = f'rgb({palette[2][0]}, {palette[2][1]}, {palette[2][2]})'  # Third color for boxes
            box_text_color = f'rgb({palette[3][0]}, {palette[3][1]}, {palette[3][2]})'    # Fourth color for box text

            # Apply CSS with dynamic color styles
            st.markdown(
                f"""
                <style>
                /* General body and app background */
                .stApp {{
                    background-color: {background_color};  /* Background based on palette color */
                    color: {text_color};  /* Text color based on palette */
                    font-family: 'Arial, sans-serif';  /* Consistent font for the whole app */
                }}

                /* Title and content alignment */
                .serif-font {{
                    font-family: 'Georgia, serif';  /* Professional serif font for headings */
                    text-align: center;
                    color: {text_color};  /* Title text color */
                }}

                /* Styled box */
                .centered-box {{
                    background-color: {box_background_color};  /* Box background dynamically set */
                    color: {box_text_color};  /* Box text color dynamically set */
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);  /* Soft shadow for depth */
                    border: 1px solid #dcdcdc;  /* Light border for structure */
                }}

                /* Seasonal palette boxes */
                .palette-box {{
                    background-color: {box_background_color};  /* Seasonal box background */
                    color: {box_text_color};  /* Seasonal box text color */
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);  /* Softer shadow */
                }}

                /* Typography in headers */
                h1, h2 {{
                    color: {text_color};  /* Dynamic headers color */
                    font-family: 'Georgia, serif';  /* Elegant font for titles */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Example of content dynamically styled based on the palette
            st.markdown(
                f'<h1 class="serif-font">Customized Color Palette-{predicted_classification} ({predicted_season})</h1>',
                unsafe_allow_html=True
            )
