from transformers import pipeline
#Zero-shot prompting
classifier=pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
sequence="I love writing code in python and building machine learning models."
labels=["sports","technology","cooking","education"]
result=classifier(sequence, candidate_labels=labels)
print(result)

from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = """
Classify the given sentence into one of the following topics:
[sports, technology, cooking, education]

Examples:
Sentence: "He played football with his friends." → sports
Sentence: "She baked a cake for her family." → cooking
Sentence: "Students learned about photosynthesis today." → education

Now classify the following sentence:
"I love eatting parotta."
Answer:
"""

result = generator(prompt, max_length=50)
print("Few-shot Result:")
print(result[0]['generated_text'])

from transformers import pipeline
prompt_cot = """
Let's reason step-by-step to classify the sentence into one of these categories:
[sports, technology, cooking, education]

Sentence: "She built a neural network to classify images."

Step 1: Identify what activity or concept the sentence talks about.
Step 2: Determine which topic it fits best.
Step 3: Give the final label.

Answer:
"""

result_cot = generator(prompt_cot, max_length=150)
print("Input: She built a neural network to classify images.")
print("Chain-of-Thought Output:\n", result_cot[0]['generated_text'])
print("\nFull Result:", result_cot)
