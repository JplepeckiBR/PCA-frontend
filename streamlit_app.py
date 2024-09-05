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
import base64


import streamlit as st


# Add custom CSS styles
st.markdown(
    """
    <style>
    /* General body and app background */
    .stApp {
        background: linear-gradient(135deg, #f0f0f0, #d9d9d9);  /* Soft gradient for the page background */
        font-family: 'Helvetica, Arial, sans-serif';  /* Minimal sans-serif font */
        color: #333333;  /* Dark grey text color */
    }

    /* Centered title and content */
    .centered-content {
        text-align: center;
    }

    /* Refined text box styling */
    .centered-box {
        background: rgba(255, 255, 255, 0.8);  /* Semi-transparent white background */
        color: #333333;  /* Dark grey text color */
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);  /* Soft shadow for subtle depth */
        border: 1px solid #e0e0e0;  /* Light border for structure */
        width: 100%; /* Full width for the content box */
    }

    /* Heading style inside the content box */
    .centered-box h1 {
        font-family: 'Helvetica, Arial, sans-serif';  /* Minimal sans-serif font for titles */
        color: #2c3e50;  /* Deep blue-grey color for titles */
        margin: 0;  /* Remove default margin */
        font-size: 3rem;  /* Larger font size for the heading */
    }

    /* Styling for the description text */
    .centered-box p {
        font-family: 'Helvetica, Arial, sans-serif';  /* Clean sans-serif font for text */
        color: #333333;
        margin-top: 10px; /* Space between title and text */
        font-size: 1rem;  /* Standard font size for the text */
    }

    /* Center the image upload button */
    .camera-input-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the description box
st.markdown(
    """
    <div class="centered-box centered-content">
        <h1>PCA</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Display the rotating rainbow-colored sphere GIF
st.image('rotating_rainbow_sphere.gif', use_column_width=True, width=400)

# Display the description box
st.markdown(
    """
    <div class="centered-box centered-content">
        <h1>Personal Colour Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# Image upload
st.markdown('<div class="camera-input-container">', unsafe_allow_html=True)
img = st.camera_input("Get Your Personal Colour Analysis!")
st.markdown('</div>', unsafe_allow_html=True)
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
