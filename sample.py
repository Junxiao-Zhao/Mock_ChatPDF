import os
import openai
import numpy as np
import pandas as pd
from MockChatPDF import *

# replace with your settings
openai.api_key = os.getenv("OPENAI_API_KEY")
file_path = r"ENTER THE PATH HERE"
question = r"ENTER THE QUESTION HERE"

# generate embeddings of a PDF
ep = EmbedPDF(file_path)
df = ep.pdf_to_df()  # convert PDF text to DataFrame
format_df = ep.format_text(df, period_type="。")  # format the DataFrame
embed_df = ep.embed(format_df)  # generate embeddings

# save and read
embed_df.to_csv("embedded.csv", index=False, encoding="utf-8-sig")
embed_df = pd.read_csv("embedded.csv", encoding="utf-8-sig")
embed_df["embeddings"] = embed_df["embeddings"].apply(eval).apply(np.array)

# ask question
chat = ChatPDF(embed_df)
answer = chat.get_answer(question=question, verbose=True)
print("Here is the answer:\n" + answer)
