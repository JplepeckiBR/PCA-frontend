import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests


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
# Example function to run the Streamlit app
def main():
    # The main content of your app would go here.
    pass

if __name__ == "__main__":
    main()

# Add some user interaction
user_input = st.text_input("Enter your name:")
if user_input:
    st.write(f"Hello {user_input}, you look slay!")


picture = st.camera_input("Take a picture")

if picture:
    st.image(picture, caption="Captured Image", use_column_width=True)
    image = Image.open(picture)


def flip_image(img_buffer):
    fastapi_url = "http://localhost:8000/flip-image/"
    files = {"file": ("filename.png", img_buffer, "image/png")}
    response = requests.post(fastapi_url, files=files)
    if response.status_code == 200:
        flipped_img_data = response.content
        flipped_img = Image.open(BytesIO(flipped_img_data))
        return flipped_img

# prep the image to send to the FastAPI
if st.button('Flip Image'):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    flipped_img = flip_image(img_buffer)
    if flipped_img:
        st.image(flipped_img, caption="Flipped Image.", use_column_width=True)
    else:
        st.error("An error occurred while processing the image.")
