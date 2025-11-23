# image_analyze.py
from load import model, processor
from PIL import Image
import torch

def analyze_image(img_path, prompt):
    # Load image
    image = Image.open(img_path).convert("RGB")

    # Build message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Remove input tokens
    trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]

    # Decode into text
    text = processor.decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return text


if __name__ == "__main__":
    img_path = r"C:\Users\rush7\OneDrive\Desktop\_run\gate_scorecard.jpg"

    question = """
    Extract: name, father_name, category, date of issue, 
    authorised_by. Return in JSON. Missing items → NA.
    """

    print("Analyzing image...")
    answer = analyze_image(img_path, question)
    print("Result:\n", answer)

