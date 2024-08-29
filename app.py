import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

def generate_image(prompt):
    try:
        image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.set_page_config(page_title="AI Image Generator", page_icon=":camera:", layout="wide")

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

prompt = st.text_input("Prompt", placeholder="Enter text here")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            image = generate_image(prompt)
            if image:
                st.image(image, caption="Generated Image")
    else:
        st.error("Please enter a prompt.")

st.markdown('<footer>Powered by Ahsan TECH</footer>', unsafe_allow_html=True)
