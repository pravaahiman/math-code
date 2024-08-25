import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app UI
st.title("Qwen-7B Text Generation")

# Text input
user_input = st.text_area("Enter your text:")

# Generate text on button click
if st.button("Generate"):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("Generated Text:")
    st.write(generated_text)
