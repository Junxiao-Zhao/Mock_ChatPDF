import pandas as pd
import openai
from openai.embeddings_utils import distances_from_embeddings


class ChatPDF:

    def __init__(self, embeded_df: pd.DataFrame) -> None:
        self.df = embeded_df

    def find_context(self, question: str, max_len: int = 1800):

        # embed the question
        embeded_q = openai.Embedding.create(
            input=question,
            engine='text-embedding-ada-002')['data'][0]['embedding']

        # get the distance from the embeddings
        self.df["dist"] = distances_from_embeddings(
            embeded_q, self.df["embeddings"].values, distance_metric='cosine')

        related_context = list()
        cur_len = 0

        # sort and find the closest context
        for row in self.df.sort_values("dist", ascending=True).iterrows():

            # count the context length
            cur_len += row["num_tokens"] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # store the related context
            related_context.append(row["text"])

        return "\n\n###\n\n".join(related_context)
    
    def get_answer()
