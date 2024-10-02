import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Replace with your Gemini API key
api_key = "AIzaSyAAEPtDR2nvNR-AetKsG3NvddGyoUJxSw4"

# Load the pre-trained model and tokenizer
model_name = "gpt-3.5-turbo "  # e.g., "gpt2-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a list of possible responses to common prompts
common_responses = [
    "I'm here to listen and support you.",
    "It's okay to feel overwhelmed. Let's talk about it.",
    "Here are some resources that might be helpful:",
    "Would you like to try a guided meditation?",
    "Let's focus on something positive. What's something you're grateful for today?"
]

# Store user preferences
user_preferences = {
    "topics": ["stress management", "mindfulness"],
    "themes": ["nature", "relaxation"],
    "meditation_style": "guided"
}

def generate_response(prompt, user_preferences):
    # Incorporate user preferences into the prompt
    personalized_prompt = f"{prompt} Consider the user's preferences: {user_preferences}"
    inputs = tokenizer(personalized_prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def get_gemini_response(prompt):
    url = "https://api.geminiai.com/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"prompt": prompt}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["text"]

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Combine responses from both models
        model_response = generate_response(user_input, user_preferences)
        gemini_response = get_gemini_response(user_input)
        combined_response = f"Model Response: {model_response}\nGemini Response: {gemini_response}"

        # Check for common prompts and provide tailored responses
        if any(keyword in user_input.lower() for keyword in ["help", "support", "advice"]):
            combined_response += f"\nHere's a suggestion: {random.choice(common_responses)}"

        return render_template('chat.html', user_input=user_input, combined_response=combined_response)
    else:
        return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)