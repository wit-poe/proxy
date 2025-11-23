Here is the full **Markdown file** you requested.
You can copy/paste it directly into `INTEGRATING_QWEN3_VL.md` or another documentation file.

---

# 🚀 Integrating Qwen3-VL-4B-Instruct Into Django + Celery Pipeline

### **Full Local GPU Inference (No HTTP Requests)**

---

## 📌 Overview

This guide explains how to integrate the **Qwen/Qwen3-VL-4B-Instruct** model directly into your Django + Celery processing pipeline.

Instead of calling a local HTTP API, your Celery worker will load the Qwen model **once** and perform inference **locally**, using your GPU for maximum speed.

This provides:

* **Zero HTTP overhead**
* **No API server required**
* **Local GPU acceleration**
* **Reliable JSON extraction**
* **Fault-tolerant retries**

---

# 1️⃣ Create a Global Qwen Loader Module

Create file:

```
your_app/vlm/local_qwen_loader.py
```

Add:

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("[QWEN] Loading Qwen3-VL-4B-Instruct model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

print("[QWEN] Model loaded successfully.")
```

### ✔ Loaded once per Celery worker

### ✔ Safe for multiprocessing

### ✔ Fast inference

---

# 2️⃣ Create the Qwen Inference Wrapper

Create:

```
your_app/vlm/qwen_inference.py
```

Add:

```python
import json
from PIL import Image
import torch
from .local_qwen_loader import model, processor


def run_qwen_extraction(image_path: str, prompt: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    # Build multimodal prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)

        trimmed = out[0][inputs["input_ids"].shape[-1]:]

        text = processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Try to parse model output as JSON
        try:
            parsed = json.loads(text)
            return {"data": parsed}
        except Exception:
            return {"error": f"Model returned non-JSON content: {text}"}

    except Exception as e:
        return {"error": f"Qwen inference error: {e}"}
```

---

# 3️⃣ Remove Old HTTP VLM Function

Delete the old function:

```python
def _call_local_vlm(...):
    ...
```

You will NOT use HTTP anymore.

---

# 4️⃣ Update Your Pipeline to Use Qwen

In your Celery task file, replace:

```python
result = _call_local_vlm(image_path, prompt)
```

with:

```python
from your_app.vlm.qwen_inference import run_qwen_extraction
result = run_qwen_extraction(image_path, prompt)
```

Everything else stays the same.

---

# 5️⃣ No Changes Needed for Retry Logic

Your retry wrapper remains fully compatible:

```python
result = extract_data_with_retry(doc.processed_file_path, prompt, expected_fields)
```

`run_qwen_extraction()` returns:

* `{"data": {...}}` on success
* `{"error": "..."}` on failure

So your existing retry logic works perfectly.

---

# 6️⃣ Optional (But Recommended) Prompt Format

Use a consistent JSON request prompt:

```
Extract the following fields from the image and return ONLY a valid JSON object:

Fields:
- name
- father_name
- category
- date_of_issue
- authorized_by

If any field is missing, return "NA".  
If text is in another language, translate values to English.  
```

---

# 7️⃣ Final Architecture Flow

```
Celery Worker
    ↓
Loads Qwen3-VL model (once)
    ↓
run_qwen_extraction()
    ↓
JSON extracted from image
    ↓
Retries if JSON incomplete
    ↓
Saved to DB + JSON artifact
```

---

# 🎯 Final Result

After integrating these steps:

✅ No local API server
✅ No HTTP requests
✅ Direct GPU inference
✅ High-speed extraction
✅ Lower latency
✅ Fewer failure points
✅ Better reliability

---

# 🤝 Need Help Expanding?

I can provide:

✔ Output validation (schema checking)
✔ Auto-correction of malformed JSON
✔ OCR-optimized prompt set
✔ Async batching for high throughput
✔ Multi-GPU support

Just ask!
