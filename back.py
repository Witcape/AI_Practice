from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Ollama LLM
model = OllamaLLM(model="llama3")

# Prompt template
template = """
You are a creative and engaging 50 words short storyteller for children aged 4-8 years. Your goal is to create fun, imaginative, and educational stories featuring animals based on the input provided.

Here is the context: The child has provided the following animal {animal} and the mood it's in is {mood}. Use the animal as the main character and describe its small adventures and emotions in detail.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# API endpoint to generate a story
@app.route('/generate_story', methods=['POST'])
def generate_story():
    try:
        data = request.json
        animal = data.get('animal')
        mood = data.get('mood')

        if not animal or not mood:
            return jsonify({'error': 'Animal and mood are required!'}), 400

        # Generate story using Llama3
        input_data = {'animal': animal, 'mood': mood}
        story = chain.invoke(input_data)

        return jsonify({'story': story.strip()})
    except Exception as e:
        return jsonify({'error': f"Something went wrong: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
