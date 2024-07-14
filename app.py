import gradio as gr
import time
import warnings
import os
from PIL import Image
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import re


import base64
import torch
from transformers import BitsAndBytesConfig, pipeline


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})


def text_from_image(input_image):
    try:
        
        if input_image is None:
            raise ValueError("Input image is None. Please provide a valid image.")

        # Loading Image
        image = Image.open(input_image)

        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, describe the disease symptoms and possible diagnosis based on the image.
        """
        prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

        # Check if 'pipe' function is callable
        if not callable(pipe):
            raise ValueError("The 'pipe' argument is not callable. Please provide a valid function.")
        
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

        # Properly extract response text
        if outputs is not None and len(outputs) > 0 and "generated_text" in outputs[0]:
            match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
            if match:
                # Extract text after "ASSISTANT:"
                reply = match.group(1)
            else:
                reply = "No response found."
        else:
            reply = "No response generated."
    
    except ValueError as ve:
        reply = str(ve)
    except Exception as e:
        reply = f"An error occurred: {str(e)}"
    
    return reply




def process_image(image_path):
    
    output = text_from_image(image_path)

    return output



interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Generated Response"),
    ],
    title="Disease Diagnosis using Llava",
    description="Upload image of disease",
    theme="default"
)

# Launch the interface
interface.launch(debug=True)