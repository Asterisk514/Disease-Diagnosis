# Disease Diagnosis Using Llava

This project leverages the Llava model for disease diagnosis by converting medical images to descriptive text. The pipeline is optimized for efficiency with 4-bit quantization, making it both powerful and resource-efficient.

## Project Overview

Medical diagnosis often relies on the interpretation of images, such as X-rays, MRIs, or photos of skin conditions. Traditionally, this requires the expertise of trained medical professionals. This project aims to assist in the diagnostic process by using an advanced neural network model to generate textual descriptions of disease symptoms and possible diagnoses based on input images.

## How It Works

1. **Image Input**: The system accepts an image related to a medical condition.
2. **Text Generation**: The Llava model, configured with 4-bit quantization for efficiency, processes the image and generates a detailed description.
3. **Output**: The generated text includes observed symptoms and potential diagnoses, aiding healthcare professionals in decision-making.

## Benefits

- **Efficiency**: Using 4-bit quantization significantly reduces the computational resources required, making it feasible to run on less powerful hardware.
- **Speed**: The pipeline provides rapid analysis, which is crucial in medical settings where time is often of the essence.
- **Support for Healthcare Professionals**: The generated descriptions can serve as a second opinion or a preliminary analysis, helping doctors and specialists in their diagnostic processes.
- **Accessibility**: This tool can be particularly useful in remote or under-resourced areas where access to specialist medical professionals is limited.

## Setup

1. **Install Dependencies**

   Ensure you have the required Python packages installed. You can do this using the following commands:
   ```bash
   pip install torch transformers nltk
   ```
