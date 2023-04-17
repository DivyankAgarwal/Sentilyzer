import streamlit as st
import torch
import torchtext
from PIL import Image
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification
import torch.nn.functional as F


# I have mixed feelings, not bad or not good.

st.set_page_config(page_title='Sentiment Analysis', page_icon=':smiley:', layout='wide')
st.sidebar.markdown('<h1 style="text-align:center; color:#D3D3D3;">Sentiment Analysis</h1>', unsafe_allow_html=True)
st.sidebar.markdown('Enter a review to classify as positive or negative.')
user_input = st.sidebar.text_input('Review')

model2 = AutoModelForSequenceClassification.from_pretrained("model_batch_20")
if model2:
    print("Model")
else:
    print("NO")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



labels = ["negative", "positive"]

st.write("<h1 style='text-align:center; color:#D3D3D3;'>Sentilyzer</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align:center; color:#D3D3D3;'>Sentiment Analysis Application</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")

with col2:
    if user_input:
        print(user_input)

        tokens = tokenizer(user_input)
        inputs = torch.tensor(tokens["input_ids"]).unsqueeze(0)

        outputs = model2(inputs)
        logits = outputs.logits

        predicted_label = torch.argmax(logits, dim=1)

        print("Predicted label: ", predicted_label)

        if predicted_label.numel() == 1:
            probs = F.softmax(logits, dim=1)
            prob_positive = probs[0][1].item()
            prob_negative = probs[0][0].item()

            prediction = predicted_label.item()
            if prediction == 1:
                st.success(f'This is a positive review with probability {prob_positive:.2f}.')
                image = Image.open("happy_image.png")
                st.image(image, use_column_width=True)
            else:
                st.error(f'This is a negative review with probability {prob_negative:.2f}.')
                image = Image.open("sad_image.png")
                st.image(image, use_column_width=True)
        else:
            st.error('Error: Model predicted more than one label.')
with col3:
    st.write("")