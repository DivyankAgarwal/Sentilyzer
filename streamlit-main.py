import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# I have mixed feelings, not bad or not good.

@st.cache_resource
def load_model():
    return AutoModelForSequenceClassification.from_pretrained("sohan-ai/test")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")



st.set_page_config(page_title='Sentiment Analysis',
                   page_icon=':smiley:',
                   layout='centered')





# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
# st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.markdown('<h1 style="text-align:center; color:#D3D3D3;">Sentiment Analysis</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<h4 style="text-align:center; color:white;">Enter a review to classify as positive or negative.</h4>', unsafe_allow_html=True)
user_input = st.sidebar.text_input('Review')

model2 = AutoModelForSequenceClassification.from_pretrained("sohan-ai/sentiment-analysis-model-amazon-reviews")
if model2:
    print("Model")
else:
    print("NO")

model2 = load_model()

tokenizer = load_tokenizer()

labels = ["negative", "positive"]

st.write("<h1 style='text-align:center; color:white;'>Sentilyzer</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align:center; color:white;'>Sentiment Analysis Application</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")

with col2:
    if user_input:

        tokens = tokenizer(user_input)
        inputs = torch.tensor(tokens["input_ids"]).unsqueeze(0)

        outputs = model2(inputs)
        logits = outputs.logits

        predicted_label = torch.argmax(logits, dim=1)


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
