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

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a creative and engaging storyteller for children aged 4-8 years. Your goal is to create fun, imaginative, and educational stories featuring animals based on the input provided.

Here is the context = The child has provided the following animal {animal} and the mood it's in is {mood}. Use the animal as the main character and his mood as an attribute in the story.
"""

model = OllamaLLM(model = "llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"animal": "tiger", "mood": "happy"})
print(result)