from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from image_processing import predict
from color_palette import extract_average_colors, predict_skin_tone_classification,predict_season,color_palette_recommendation_and_visualization


app = FastAPI()
@app.get("/")
def root():
    return "Hello World"


@app.post("/flip-image")
async def flip_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    img_io = io.BytesIO() # Create an in-memory bytes buffer.
    flipped_image.save(img_io, 'PNG') # Save the flipped image to the buffer in JPEG format.
    img_io.seek(0)# Move the file pointer to the beginning of the buffer.
    return StreamingResponse(img_io, media_type="image/png") #officially type for JPEG images




#Type Annotation (file: UploadFile):Tells FastAPI that this parameter should be treated as an uploaded file.
#Enables the use of UploadFile methods and attributes, such as file.file, file.filename, file.content_type, etc.

@app.post("/get_masks")
async def flip_image(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img_io = io.BytesIO()
    labels = predict(img)
    print(labels)
    # TODO check the data type of returning the prediction
    if labels is not None:
            return {"labels": str(labels.tolist()), "label_type": str(type(labels))}
    else:
            return {"error": "Prediction returned None", "label_type": None}




@app.post("/Get recommended palette")
async def flip_and_process_image(file: UploadFile = File(...)):
    # Step 1: Open and flip the image
    image = Image.open(file.file)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Step 2: Extract average colors from the flipped image
    average_skin_color, average_brows_color, average_hair_color, average_lip_color, average_eye_color = extract_average_colors(flipped_image)

    # Step 3: Predict skin tone classification
    predicted_skin_tone = predict_skin_tone_classification(average_skin_color)

    # Step 4: Predict season
    predicted_season = predict_season(
        average_eye_color,
        average_hair_color,
        average_lip_color,
        average_brows_color
    )

    # Step 5: Get color palette recommendations and visualization
    recommendations = color_palette_recommendation_and_visualization(predicted_skin_tone, predicted_season)



   # Step 6: Return only the color palette from recommendations
    return {'color_palette': recommendations['palette']}
