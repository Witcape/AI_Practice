# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.78 numpy==1.23.4 --force-reinstall --upgrade --no-cache-dir --verbose
# pip install huggingface_hub
# pip install llama-cpp-python==0.1.78
# pip install numpy==1.23.4

# CostFunction = sphere  #
# nVar = 10  # Number of Decision Variables
# VarSize = (nVar,)  # Size of Decision Variables Matrix
# VarMin = -10  # Lower Bound of Variables
# VarMax = 10  # Upper Bound of Variables

# w = 1  # Inertia Weight
# wdamp = 0.99  # Inertia Weight Damping Ratio
# c1 = 1.5  # Personal Learning Coefficient
# c2 = 2.0  # Global Learning Coefficient

# particles = []

# for _ in range(nPop):
#     position = np.random.uniform(VarMin, VarMax, VarSize)
#     velocity = np.zeros(VarSize)
#     cost = CostFunction(position)
#     best_position = position.copy()
#     best_cost = cost
#     particles.append({
#         'Position': position,
#         'Velocity': velocity,
#         'Cost': cost,
#         'Best': {
#             'Position': best_position,
#             'Cost': best_cost
#         }
#     })
# print(np.size(particles))

# GlobalBest = {'Position': None, 'Cost': np.inf}

# for p in particles:
#     if p['Cost'] < GlobalBest['Cost']:
#         GlobalBest = {'Position': p['Best']['Position'].copy(), 'Cost': p['Best']['Cost']}

# BestCost = np.zeros(MaxIt)
     
from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Ollama LLM
model = OllamaLLM(model="llama3")

template = """
You are a creative and engaging teacher for students aged 16 - 30 years. Your goal is to explain topics in an educational manner. Use relatable examples to make the concepts engaging and easy to understand.

Here is the context: The student has provided the following topic: {topic}. the question is : {question}. Avoid answering questions that don't align with {topic}.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# API endpoint to generate a prompt based on topic and question
@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    try:
        data = request.json
        topic = data.get('topic')
        question = data.get('question')

        if not topic or not question:
            return jsonify({'error': 'Topic and question are required!'}), 400

        # Generate response based on the topic and question
        input_data = {'topic': topic, 'question': question}
        prompt_response = chain.invoke(input_data)

        return jsonify({'prompt': prompt_response.strip()})
    except Exception as e:
        return jsonify({'error': f"Something went wrong: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
