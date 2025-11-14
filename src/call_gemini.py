from google import genai
from google.genai import types
from google.api_core import retry
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable,
    )(genai.models.Models.generate_content)

config = types.GenerateContentConfig(temperature=0.0)
client = genai.Client(api_key=api_key)
chat = client.chats.create(model='gemini-2.0-flash')

def call_gemini(prompt):
    try:
        response = chat.send_message(
            message = prompt
        )
        text = response.text.strip()
        print("Gemini response: ",text)

        try:
            return text
        except Exception as e:
            print(f"Model error: {e}", "error")
    except Exception as e:
        print(f"An error occurred: {e}", "error")
        return None
    
    