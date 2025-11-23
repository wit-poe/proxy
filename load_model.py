import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- Configuration ---
# The model identifier on Hugging Face Hub
MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'

# The name of the local directory where the model will be saved.
# This will be created in the same folder where you run this script.
SAVE_DIRECTORY = './Qwen/Qwen3-VL-4B-Instruct'


def download_and_save_model():
    """
    Downloads a model from Hugging Face and saves it to a local directory.
    """
    if os.path.exists(SAVE_DIRECTORY):
        print(f"Directory '{SAVE_DIRECTORY}' already exists. Skipping download.")
        print("If you want to re-download, please delete this folder first.")
        return

    print(f"Starting download of model: {MODEL_ID}")
    print(f"This is a large model (~16 GB) and will take a significant amount of time and disk space.")

    try:
        # Download the model and processor from the hub
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)

        # Save the downloaded files to the specified local directory
        print(f"Saving model files to '{SAVE_DIRECTORY}'...")
        processor.save_pretrained(SAVE_DIRECTORY)
        model.save_pretrained(SAVE_DIRECTORY)

        print("\n-------------------------------------------------")
        print("Success! Model downloaded and saved successfully.")
        print(f"You can now point your inference script to: {SAVE_DIRECTORY}")
        print("-------------------------------------------------")

    except Exception as e:
        print(f"\nAn error occurred during download or saving: {e}")
        print("Please check your internet connection and available disk space.")

if __name__ == "__main__":
    download_and_save_model()