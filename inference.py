import streamlit as st
import torch
from transformers import  AutoTokenizer,T5ForConditionalGeneration
import nltk
from nltk import FreqDist
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import json
import numpy as np
import random
from PIL import Image

#
image_path = "image.png"
image = Image.open(image_path)

def preprocess_function(text_path,content_type = None ):
    #with open(text_path,"r") as f:
    #  data = f.read()
    print(text_path)
    sentences = [sent_tokenize(text_path)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    joiner = " "
    modified_text = joiner.join(sentences)
    answer = bool(random.choice([0,1]))
    form = "truefalse: %s passage: %s </s>" % (modified_text, answer)
    print(form)
    return (form,answer)

# 
def predict_function(data,Model): 
    print(data)
    text,answer = data
    if answer == True:
        ans = 'Yes'
    else:
        ans = 'No'
    tokenizer, model  = Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = model.generate(input_ids=input_ids,
                            attention_mask=attention_masks,
                            max_length=256,
                            num_beams=10,
                            num_return_sequences=3,
                            no_repeat_ngram_size=2,
                            early_stopping=True
                            )
    Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in output]
    result = [Question.strip().capitalize() for Question in Questions]
    final = pd.DataFrame({'Questions':result,'Answer':[ans]*len(result)})
    return final
#
@st.experimental_singleton
def model_load_function(model_path=None):
    print("loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
    print("model was loaded")
    return (tokenizer,model.to(device))
#
st.image(image, caption='Boolean Question Generator')
st.write("""## ?????? Boolean Question Generator App ??????""")

query = st.text_input("Enter the Context here.", "")

if query != "":
    with st.spinner(text="Initializing the Model...It might take a while..."):
        model = model_load_function(model_path=None)
    with st.spinner(text="Preprocessing the Context...It might take a while..."):
        data = preprocess_function(query)
    with st.spinner(text="Making Predictions...It might take a while..."):    
        predictions = predict_function(data,model)

    with st.spinner(text="Questions Generated ????????????"):
        st.dataframe(predictions)