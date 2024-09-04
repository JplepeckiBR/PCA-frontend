import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import json
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from image_processing import predict
from colour_palette import extract_average_colors, predict_skin_tone_classification ,predict_season,color_palette_recommendation_and_visualization
import cv2



# Set custom CSS for font style, centering, and text box styling
st.markdown(
    """
    <style>
    .serif-font {
        font-family: 'serif';
        text-align: center;
    }
    .stApp {
        background-color: #f5f5f5;  /* Optional: You can set a background color or use a background image as well */
    }
    .centered-box {
        background-color: #B3E5FC;
        color: #0D47A1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .summer-palette-box {
        background-image: url("https://t3.ftcdn.net/jpg/02/78/47/12/360_F_278471261_orsEBJmDEQ2RZKknNxYwQ56i9bqwizlM.jpg");
        background-size: cover;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .winter-palette-box {
        background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20230611/pngtree-snowy-landscape-wallpaper-with-trees-snow-white-winter-image_2935694.jpg");
        background-size: cover;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .fall-palette-box {
        background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20230613/pngtree-fall-wallpaper-autumn-autumn-wallpaper-image_2945745.jpg");
        background-size: cover;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    .spring-palette-box {
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/041/262/808/small_2x/ai-generated-blossoming-pink-cherry-trees-in-vibrant-spring-meadow-photo.jpg");
        background-size: cover;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'serif';
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title
st.markdown(
    '<h1 class="serif-font" style="color: black;">🌈 Personal Recommendation System 🌈</h1>',
    unsafe_allow_html=True
)

# Description
st.markdown(
    '<p class="serif-font" style="color: black;">Get a personal color analysis to find out what products best suit you!</p>',
    unsafe_allow_html=True
)

# Additional text box with detailed description, centered and styled
st.markdown(
    '<div class="centered-box">This color analysis tool uses advanced face segmentation technology to assess your unique features. By analyzing a photo of you, the code meticulously examines various aspects of your face, including your eyes, hair, eyebrows, lips, and skin tone. These elements are carefully evaluated to determine which color palette complements your natural appearance best. The process involves identifying the specific shades and tones within each facial feature, ensuring a personalized recommendation of colors that enhance your overall look.</div>',
    unsafe_allow_html=True
)

# Text box with information about color palettes
st.markdown(
    '<div class="centered-box">Color palettes are often categorized by seasonal color theory, which associates different tones and hues with the four seasons: Summer, Winter, Fall (Autumn), and Spring. Each season has its own unique palette, further divided into subcategories like Soft, Light, Warm, Cool, Deep, and Clear. Here\'s a detailed description:</div>',
    unsafe_allow_html=True
)

# Summer Palette text box with custom background
st.markdown(
    """
    <div class="summer-palette-box">
    <h2>Summer Palette</h2>
    <p><strong>Soft Summer:</strong> A muted and cool palette with a soft, gentle feel. Colors include dusty rose, muted lavender, and soft teal. The overall effect is subtle and understated.</p>
    <p><strong>Light Summer:</strong> A light, airy palette with cool undertones. Colors are delicate and pastel-like, such as baby blue, light pink, and mint green. This palette is fresh and breezy.</p>
    <p><strong>Cool Summer:</strong> A cool and serene palette with more intensity than the Soft Summer. It includes colors like powder blue, lavender, and soft navy. The overall feel is calm and refined.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Winter Palette text box with custom background
st.markdown(
    """
    <div class="winter-palette-box">
    <h2>Winter Palette</h2>
    <p><strong>Cool Winter:</strong> A sharp, cool palette with high contrast. Colors include icy blues, pure white, and deep black. This palette is striking and crisp.</p>
    <p><strong>Deep Winter:</strong> A rich, bold palette with deep and cool tones. It features colors like deep burgundy, forest green, and midnight blue. The effect is dramatic and intense.</p>
    <p><strong>Clear Winter:</strong> A bright, high-contrast palette with cool undertones. Colors include vivid red, bright emerald, and stark black. This palette is bold and eye-catching.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Fall (Autumn) Palette text box with custom background
st.markdown(
    """
    <div class="fall-palette-box">
    <h2>Fall (Autumn) Palette</h2>
    <p><strong>Warm Fall:</strong> A warm, earthy palette with rich, golden undertones. It includes colors like burnt orange, mustard yellow, and olive green. This palette is cozy and inviting.</p>
    <p><strong>Deep Fall:</strong> A deep, warm palette with intense and rich tones. Colors include dark chocolate, deep rust, and forest green. The overall effect is grounded and intense.</p>
    <p><strong>Soft Fall:</strong> A muted, warm palette with gentle, earthy tones. It features colors like taupe, soft olive, and muted terracotta. This palette is subtle and natural.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Spring Palette text box with custom background
st.markdown(
    """
    <div class="spring-palette-box">
    <h2>Spring Palette</h2>
    <p><strong>Light Spring:</strong> A light, warm palette with fresh, pastel-like colors. It includes peach, light coral, and soft yellow. The overall feel is delicate and cheerful.</p>
    <p><strong>Warm Spring:</strong> A warm, vibrant palette with golden undertones. Colors include bright coral, golden yellow, and warm turquoise. This palette is lively and energetic.</p>
    <p><strong>Clear Spring:</strong> A bright, clear palette with warm undertones. It features colors like vivid orange, bright green, and clear blue. The effect is fresh and radiant.</p>
    </div>
    """,
    unsafe_allow_html=True
)


    # def flip_image(img_buffer):
    #     fastapi_url = "http://localhost:8000/flip-image/"
    #     files = {"file": ("filename.png", img_buffer, "image/png")}
    #     # files = {'image': open(file_path, 'rb')}
    #     response = requests.post(fastapi_url, files=files)

    #     if response.status_code == 200:
    #         flipped_img_data = response.content
    #         flipped_img = Image.open(BytesIO(flipped_img_data))

    #         return flipped_img_data

    # # prep the image to send to the FastAPI
    # if st.button('Flip Image'):
    #     img_buffer = BytesIO()
    #     image.save(img_buffer, format="PNG")
    #     img_buffer.seek(0)

    #     flipped_img = flip_image(img_buffer)


# def main():
#     st.title("Face Parsing with Segformer")

#     # Image upload
#     img = st.camera_input("Take a picture")

#     if img is not None:
#         # Load image
#         image = Image.open(img)

#         # Display the original image
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Run the prediction
#         st.write("Running inference...")
#         labels_viz = predict(image)

#         # Display the segmentation mask (optional: apply a color map)
#         st.image(labels_viz, caption='Segmentation Mask', use_column_width=True)

# if __name__ == "__main__":
#     main()




st.title("Face Parsing with Segformer")

# Image upload
img = st.camera_input("Take a picture")

if img is not None:
# Load image
        image = Image.open(img)

        # Display the original image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Show mask + Extract average colors
        st.write("Running inference...")
        skin,average_skin_color, average_brows_color, average_hair_color, average_lip_color, average_eye_color = extract_average_colors(image)

        #predict warm neutral or cool
        predicted_classification = predict_skin_tone_classification(extract_average_colors(image)[1])

        #predict which season
        predicted_season = predict_season(average_eye_color,average_hair_color, average_lip_color, average_brows_color )

        #output palette
        color_palette = color_palette_recommendation_and_visualization(predicted_classification,predicted_season)

        # Display the segmentation mask (optional: apply a color map)
        colors = [
            (average_skin_color, "skin"),
            (average_brows_color, "brows"),
            (average_hair_color, "hair"),
            (average_eye_color, "eyes"),
            (average_lip_color, "lips")
        ]

        # fig = plt.subplots(1, len(colors), figsize(15,3))

        for (color, title) in colors:
            image = np.ones((100,100,3))
            image[:,:,0] *= color[0] / 255.0
            image[:,:,1] *= color[1] / 255.0
            image[:,:,2] *= color[2] / 255.0

            st.image(image)
            st.write(title)




        st.image(average_skin_color, caption='facial features colors', use_column_width=True)
        st.image(color_palette, caption='your personalized color palette', use_column_width=True)
