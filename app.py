import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Initialize the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Image generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Streamlit application
st.set_page_config(page_title="AI Image Generator", page_icon=":camera:", layout="wide")

# Custom CSS for beautifying the interface
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        border-radius: 15px;
        padding: 20px;
    }
    footer {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: #4CAF50;
    }
    .stTextInput, .stTextArea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Image Generator")
st.markdown("**Enter a prompt to generate an image**")

# Input and Output
prompt = st.text_input("Prompt", placeholder="Enter text here")
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            image = generate_image(prompt)
            st.image(image, caption="Generated Image")
    else:
        st.error("Please enter a prompt.")

# Footer
st.markdown('<footer>Powered by Ahsan TECH</footer>', unsafe_allow_html=True)
