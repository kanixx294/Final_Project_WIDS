import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="CinemaChat", layout="centered")
st.title("🎥 CinemaChat AI")

@st.cache_resource
def get_bot():
    path = "./final_model"
    return AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path)

tokenizer, model = get_bot()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask me anything..."):
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')
        outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id, 
                                 do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
        
        reply = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        st.write(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})