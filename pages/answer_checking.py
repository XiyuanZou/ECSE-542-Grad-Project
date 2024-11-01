import streamlit as st
import os, pickle
from utils import self_checking

# Function to highlight similar key points
def highlight_similar(idx):
    st.session_state.highlighted = similar_points_map[idx]+[idx]
        
        
models = ["gpt-3.5-turbo-1106", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"]
with open("document_text", "rb") as fp: 
    document_text =pickle.load(fp)
with open("summary_text_lst", "rb") as fp: 
    summary_text_lst=pickle.load(fp)
with open("similar_points_map", "rb") as fp: 
    similar_points_map=pickle.load(fp)
    
#get the confidence score of all key points
confidence_map={}
for target_key_point_idx in range(len(similar_points_map)):
    num_confident_models = 0
    kp_index = 0
    
    for s_idx, summary in enumerate(summary_text_lst):
        confident=False
        for kp in summary:
            if kp in [" ", "\n", ""]: #ignore padding points
                continue
            if kp_index in similar_points_map[target_key_point_idx]+[target_key_point_idx]:
                confident=True
                break
            kp_index += 1
        if(confident):
            num_confident_models+=1
    
    confidence_map[target_key_point_idx]=num_confident_models

    

# Store key points in session state to keep track of highlights
if 'highlighted' not in st.session_state:
    st.session_state.highlighted = []

kp_index = 0
for s_idx, summary in enumerate(summary_text_lst):
    st.markdown(f"# **{models[s_idx]}**")
    for kp in summary:
        if kp in [" ", "\n", ""]: #ignore padding points
            continue

        is_highlighted = kp_index in st.session_state.highlighted # Check if the current key point should be highlighted
        color = "blue" if is_highlighted else "black"
        
        # Display key point with color and button
        st.markdown(
            f"<div style='background-color: {color}; padding: 5px; border-radius: 5px;'>{kp}</div>",
            unsafe_allow_html=True
        )
        if st.button("Cross-model Checking", key=f"cross_btn_{kp_index}", on_click=highlight_similar, args=(kp_index,)):
            st.markdown(f"Cross-model confidence: {confidence_map[kp_index]}/{len(models)}")
        if st.button("Self Checking", key=f"self_btn_{kp_index}"):
            self_confidence=self_checking(models[s_idx], kp, document_text)
            st.markdown(f"Self-checking confidence: {self_confidence}/10")
        st.markdown("  \n")
        
        kp_index += 1