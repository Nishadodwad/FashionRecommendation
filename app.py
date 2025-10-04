# ---------- app.py ----------
import streamlit as st
import torch, clip, faiss, requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch.nn.functional as F
import re, os

# ------------------------------
# Utility functions
# ------------------------------
@st.cache_resource
def load_model_and_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load embeddings & index
    emb_dir = "embeddings"
    train_img_embs = np.load(os.path.join(emb_dir, "train_image.npy"))
    train_txt_embs = np.load(os.path.join(emb_dir, "train_text.npy"))
    index = faiss.IndexFlatIP(train_img_embs.shape[1])
    index.add(train_img_embs.astype('float32'))

    return model, preprocess, device, train_img_embs, train_txt_embs, index

def clean_text(text, max_len=400):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:max_len]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("🧥 Multimodal Fashion Recommendation System")

model, preprocess, device, train_img_embs, train_txt_embs, index = load_model_and_index()

# Load dataset (must contain img_url & name)
import pandas as pd
train_df = pd.read_csv("train_df.csv")  # save your train_df earlier in Colab

choice = st.radio("Choose query type:", ["Text Query", "Image Upload"])

# -------------- TEXT QUERY --------------
if choice == "Text Query":
    query_text = st.text_input("Enter a product description:")
    if st.button("Search") and query_text.strip():
        with torch.no_grad():
            token = clip.tokenize([clean_text(query_text)], truncate=True).to(device)
            q_emb = model.encode_text(token)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
            q_emb = q_emb.cpu().numpy().astype('float32')

        D, I = index.search(q_emb, 6)
        st.subheader("Top Matches:")
        cols = st.columns(3)
        for rank, idx in enumerate(I[0][:6]):
            name = train_df.iloc[idx]['name']
            url = train_df.iloc[idx]['img_url']
            try:
                image = Image.open(requests.get(url, stream=True, timeout=10).raw)
                cols[rank % 3].image(image, caption=name, use_column_width=True)
            except:
                cols[rank % 3].write(name)

# -------------- IMAGE QUERY --------------
if choice == "Image Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded and st.button("Find Similar"):
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", width=250)
        with torch.no_grad():
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            q_emb = model.encode_image(img_tensor)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
            q_emb = q_emb.cpu().numpy().astype('float32')

        D, I = index.search(q_emb, 6)
        st.subheader("Top Matches:")
        cols = st.columns(3)
        for rank, idx in enumerate(I[0][:6]):
            name = train_df.iloc[idx]['name']
            url = train_df.iloc[idx]['img_url']
            try:
                image = Image.open(requests.get(url, stream=True, timeout=10).raw)
                cols[rank % 3].image(image, caption=name, use_column_width=True)
            except:
                cols[rank % 3].write(name)
