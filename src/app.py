# import the require libraries
import gradio as gr  # for UI interface
import base64  # for optimizing the image for the model
from openai import OpenAI  # accesing open api
import os  # for interacting with the operating system
from dotenv import load_dotenv  # for handiling environemt variables
from IPython.display import Markdown, display


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check the key

if not api_key:
    print(
        "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
    )
elif not api_key.startswith("sk-proj-"):
    print(
        "An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook"
    )
elif api_key.strip() != api_key:
    print(
        "An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook"
    )
else:
    print("API key found and looks good so far!")
# Initialize OpenAI client
client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to analyze image
def analyze_image(image):
    # Encode the image in base64
    base64_image = encode_image(image)

    # System prompt
    system_prompt = """
    You are an expert assistant trained to analyze car damage for insurance claims. 
    Your task is to assess images of cars and classify the damage as 'Minor', 'Moderate', or 'Severe', providing reasoning for your classification."

    **Role:** Insurance Officer for Automotive Damage Assessment

    **Objective:**
    1. Analyze images to determine if the car is damaged or not.
    2. If damaged, classify the severity as one of the following:
    - Minor: Cosmetic damage with little to no impact on functionality.
    - Moderate: Noticeable damage that may require repair but doesn’t affect the car’s structural integrity.
    - Severe: Significant damage affecting the car’s functionality, safety, or structure, requiring extensive repair.

    **Context:**
    This system helps automate the insurance claims process for faster and more accurate damage assessments. It will process high-quality images to improve accuracy in determining damage severity and affect claim approval timelines and customer satisfaction.

    **Instructions:**
    - **Input:** Images of cars taken from multiple angles and under different lighting conditions.
    - **Output:** 
    1. Binary classification: **Damaged** or **Not Damaged**
    2. If damaged, classify severity into Minor, Moderate, or Severe with a concise explanation for each classification.

    **Considerations:**
    - Handle variations in lighting, angles, and backgrounds effectively.
    - Identify common types of car damage, such as scratches, dents, cracks, and broken parts.
    - Focus on precision and accuracy in determining the damage severity.

    **Additional Notes:**
    - **Reasoning:** Provide clear and concise reasoning for your predictions, ensuring that the explanation aligns with the visible damage in the image.
    - **Accuracy & Precision:** Prioritize high accuracy and precision in both the detection of damage and categorization of severity.

    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the severity of the car damage? Please classify as Minor, Moderate, or Severe with reasoning.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        temperature=0.2,
        top_p=1.0,
        max_tokens=100,
    )

    # Extract the response
    if response:
        result = response.choices[0].message.content
        return result
    else:
        return "Failed to analyze the image. Please try again."


# Gradio interface
interface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Car Damage Severity Analyzer",
    description="Upload an image of car damage to analyze the severity as Minor, Moderate, or Severe.",
)

interface.launch()
