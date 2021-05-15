#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer


# In[3]:


dataset = pd.read_excel(r'C:/Users/rayad/Downloads/Chatbot_Dep/Chatbot_upd.xlsx', engine='openpyxl')


# In[4]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
dataset.tail()


# In[5]:


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
    return model([input])

dataset['Question_Vector'] = dataset.Questions.map(embed)
dataset['Question_Vector'] = dataset.Question_Vector.map(np.array)
pickle.dump(dataset, open('chatdata.pkl', 'wb'))


# In[6]:


model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
dataset = pickle.load(open('chatdata.pkl', mode='rb'))
questions = dataset.Questions
QUESTION_VECTORS = np.array(dataset.Question_Vector)
COSINE_THRESHOLD = 0.5


# In[7]:


def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)


# In[8]:


def semantic_search(query, data, vectors):        
    query_vec = np.array(embed(query))
    res = []
    for i, d in enumerate(data):
        qvec = vectors[i].ravel()
        sim = cosine_similarity(query_vec, qvec)
        res.append((sim, d[:100], i))
    return sorted(res, key=lambda x : x[0], reverse=True)


# In[9]:


def generate_answer(question):
    '''This will return list of all questions according to their similarity,but we'll pick topmost/most relevant question'''
    most_relevant_row = semantic_search(question, questions, QUESTION_VECTORS)[0]
#     print(most_relevant_row)
    if most_relevant_row[0][0]>=COSINE_THRESHOLD:
        answer = dataset.Answers[most_relevant_row[2]]
        return answer
    else:
        no_answer = "Sorry I am not able to get you!"
    return no_answer
    


# In[10]:


question_1 = 'Hello'


# In[11]:


generate_answer(question_1)


# In[12]:


question_2 = 'what is machine learning?'


# In[13]:


generate_answer(question_2)


# In[ ]:


question_3 = 'what is SVM'


# In[ ]:


generate_answer(question_3)


# In[ ]:




