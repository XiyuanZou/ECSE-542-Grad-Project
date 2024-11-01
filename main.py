from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def get_summary(content, model_name):
    system_message = [
        {
            "role": "system",
            "content": "Summarize the following document. List each key point in its own paragraph. Add [end] at the end of each key point as a separator.",
        }
    ]
    
    file_message = [{
        "role": "user",
        "content": content,
    }]
    
    try:
        response = client.chat.completions.create(model=model_name, messages=system_message+file_message)  # type: ignore
        return response.choices[0].message.content
    except:
        return "Failed to make a summary. Please retry!"
    
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    try:
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    except:
        return [0]*3072



st.title("Multi-LLM Document Summarizer")
st.subheader("with Self Checking and Cross-model Checking")

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_KEY"])
models = ["gpt-3.5-turbo-1106", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"]
SIMILARITY_THRESHOLD = 0.85

file = st.file_uploader("Upload your document", type="pdf")

if file is not None:
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(file)
    # Extract the content
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[page].extract_text()
    
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            summary_text_lst=[] # (num_models, num_key_points)
            summary_embedding_lst=[] # (num_models*num_key_points, embed_dim)
            
            for model_name in models:
                st.markdown("---")
                llm_summary = (
                    get_summary(content, model_name)
                    or "No response given!"
                )
                key_points_lst=llm_summary.split("[end]")
                summary_text_lst.append(key_points_lst[:-1])
                st.markdown(f"# **{model_name}**  \n {llm_summary}")      
            
            max_num_key_points=0
            for key_points_lst in summary_text_lst:
                if(len(key_points_lst)>max_num_key_points):
                    max_num_key_points=len(key_points_lst)
                    
            for key_points_lst in summary_text_lst:
                key_points_lst.extend([" "]*(max_num_key_points-len(key_points_lst))) #padding the key points lst of each summary
                key_points_embedding_lst=[get_embedding(key_point) for key_point in key_points_lst]
                summary_embedding_lst.extend(key_points_embedding_lst)
            
            similarity_matrix = cosine_similarity(summary_embedding_lst) # similarity_matrix (num_models*num_key_points, num_models*num_key_points)
            
            # Create a dictionary to store high-similarity mappings
            similar_points_map = {}
            key_points = [kp for summary in summary_text_lst for kp in summary]
            for i, _ in enumerate(key_points):
                similar_points_map[i] = [j for j in range(len(key_points)) if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD and i != j] #similarity_points_map (num_models*num_key_points, )

            with open("document_text", "wb") as fp: 
                pickle.dump(content, fp)
            with open("summary_text_lst", "wb") as fp: 
                pickle.dump(summary_text_lst, fp)
            with open("similar_points_map", "wb") as fp: 
                pickle.dump(similar_points_map, fp)
                
            if st.button("Begin Answer Checking", key="answer checking button"):
                st.switch_page("pages/answer_checking.py")
            
            print(similar_points_map)

    
