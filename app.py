from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
 
 
    
app = Flask(__name__)

dataset = pd.read_excel(r'Chatbot.xlsx', engine='openpyxl')

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
    return model([input])

dataset['Question_Vector'] = dataset.Questions.map(embed)
dataset['Question_Vector'] = dataset.Question_Vector.map(np.array)
pickle.dump(dataset, open('chatdata.pkl', 'wb'))

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
dataset = pickle.load(open('chatdata.pkl', mode='rb'))
questions = dataset.Questions
QUESTION_VECTORS = np.array(dataset.Question_Vector)
COSINE_THRESHOLD = 0.5

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def semantic_search(query, data, vectors):        
    query_vec = np.array(embed(query))
    res = []
    for i, d in enumerate(data):
        qvec = vectors[i].ravel()
        sim = cosine_similarity(query_vec, qvec)
        res.append((sim, d[:100], i))
    return sorted(res, key=lambda x : x[0], reverse=True)

def generate_answer(question):
    most_relevant_row = semantic_search(question, questions, QUESTION_VECTORS)[0]
#     print(most_relevant_row)
    if most_relevant_row[0][0]>=COSINE_THRESHOLD:
        answer = dataset.Answers[most_relevant_row[2]]
        return answer
    else:
        no_answer = "Sorry I am not able to get you!"
    return no_answer
    


@app.route("/")
def home():
    return render_template("page.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(generate_answer(userText))
 

if __name__ == "__main__":
    app.run()
