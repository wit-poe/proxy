# ai_service/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
import base64
import logging
import uvicorn

# --- CONFIGURATION ---
# We use a local path if available (for air-gapped setups), or the Hub ID
# You can change this to the absolute path where you downloaded the model
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" 

# Determine device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Running on NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Running on Apple Silicon (MPS)")
else:
    DEVICE = "cpu"
    print("⚠️ Running on CPU (Slow)")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QwenService")

app = FastAPI()

# Global variables
model = None
processor = None

# --- DATA MODELS ---
class ExtractionRequest(BaseModel):
    prompt: str
    image_base64: str # We expect the image as a base64 string

# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    global model, processor
    logger.info(f"Loading Qwen Model on {DEVICE}...")
    
    try:
        # 1. Load Model
        # Your load.py logic goes here.
        # trust_remote_code=True is often needed for Qwen
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype="auto",
            device_map="auto", 
            trust_remote_code=True
        )
        
        # 2. Load Processor
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        logger.info("✅ Qwen Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

# --- API ENDPOINT ---
@app.post("/generate")
async def generate_text(request: ExtractionRequest):
    """
    Receives an image (base64) and a prompt. Returns the model's text output.
    """
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. Decode Base64 Image
        try:
            # Handle data URI scheme (e.g. "data:image/jpeg;base64,.....")
            if "," in request.image_base64:
                base64_data = request.image_base64.split(",")[1]
            else:
                base64_data = request.image_base64
                
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # 2. Build Messages (Matching your test.py logic)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, # PIL Image object
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]

        # 3. Prepare Inputs using Qwen's utility
        text_inputs = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_inputs],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the correct device
        inputs = inputs.to(model.device)

        # 4. Generate
        # We wrap this in no_grad for efficiency
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        # 5. Decode Output (Trim input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Return the result
        # output_text is a list, we take the first one
        return {"text": output_text[0]}

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)