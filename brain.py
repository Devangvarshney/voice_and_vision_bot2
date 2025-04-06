import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# convert image to required formate

import base64

image_path = "acen.jpg"


def encoded_image(image_path):
    image_file = open(image_path, "rb")
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


# step 3= setup  llm model
from groq import Groq

#query = "is there something wrong on my face ? tell detail and what are the precaution we are used in this situations?"
model = "llama-3.2-90b-vision-preview"


def analyze_image_with_query(query, model, encoded_image):
    client = Groq()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return (chat_completion.choices[0].message.content)

