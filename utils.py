from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_KEY"])

def self_checking(model, key_point, document_text):
    system_message = [
        {
            "role": "system",
            "content": "Give a confidence score that the following point is a key point of the given document important enough to be included in its summary. Give your confidence as a number out of 10. Only show the final number in your answer.",
        }
    ]
    
    user_message = [{
        "role": "user",
        "content": "Point:\n"+key_point+"Given document:\n"+document_text,
        }       
    ]
    
    try:
        response = client.chat.completions.create(model=model, messages=system_message+user_message)  # type: ignore
        return response.choices[0].message.content
    except:
        return 5