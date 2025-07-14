
import torch
import time
from PIL import Image as PilImage, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import io

# --- Model and Processor Loading ---
# It's best practice to load the model once when the application starts.
model_name = "fhswf/TrOCR_Math_handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()

# --- Core OCR and Preprocessing Functions (from your notebook) ---

def run_mathOCR(image, measure_time: bool = False):
    """
    Converts an image to text using mathOCR.
    """
    if measure_time:
        t0 = time.perf_counter()

    # Process image to pixel values and move to device
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate text IDs
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True
    )

    # Decode generated IDs to text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if measure_time:
        t1 = time.perf_counter()
        print(f"[Timer] Elapsed: {t1 - t0:0.4f} sec")

    return generated_text

def preprocess_image(image_bytes, target_size=(384, 384)):
    """
    Preprocesses the image from bytes.
    """
    img = PilImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.pad(img, target_size, color="white")
    return img

# --- API Endpoints ---

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, runs OCR, and returns the LaTeX result.
    """
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    extracted_text = run_mathOCR(processed_image)
    return {"latex": extracted_text}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # In a real application, you would serve index.html from a file.
    # For simplicity, we are embedding it here.
    # I will create a separate index.html file next.
    return HTMLResponse(content="<h1>Go to /docs for the API documentation or use a frontend to call the /ocr/ endpoint.</h1>")
