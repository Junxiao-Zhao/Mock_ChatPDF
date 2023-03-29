import pandas as pd
import pdfplumber
import os
import re
import tiktoken
import openai
import time


class EmbedPDF:

    def __init__(self, file_path: str) -> None:
        """ This is a class for converting PDF to a DataFrame with embeddings of the text

        :param file_path: the path of the PDF
        """

        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        self.contents = list()
        self.df = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def pdf_to_df(self, **kwargs) -> pd.DataFrame:
        """ Convert PDF text to pandas DataFrame

        :param kwargs: parameters for pdfplumber.open
        :return: a DataFrame contains Page No. and Contents
        """

        print("Reading pdf...")
        with pdfplumber.open(self.file_path, **kwargs) as pdf:
            self.pages = pdf.pages

            for i, page in enumerate(self.pages):
                print(f"Reading Page {i+1}...")
                txt = re.sub(r'\s', ' ', page.extract_text())
                self.contents.append([i, txt])

        print("Finish reading!\n")

        self.contents[0][1] = self.file_name.split(
            ".")[0] + " " + self.contents[0][1]  # add file name to Page 0

        return pd.DataFrame(self.contents, columns=("Page No.", "Contents"))

    def break_long_text(self, text: str, **kwargs) -> list:
        """ Break the long text into chunks of a maximum number of tokens

        :param text: the long text
        :param kwargs: could contain period_type: str and max_token: int
        :return: a list of chunks
        """

        period_type = kwargs.get(
            "period_type")  # default using . to split text
        max_token = kwargs.get("max_token")

        sentences = text.split(period_type)
        chunks = list()
        count_tokens = 0
        to_merge = list()

        num_tokens = [(s, len(self.tokenizer.encode(s))) for s in sentences]

        for s, n in num_tokens:

            # ignore single sentence if its tokens exceed the maximum
            if n > max_token:
                continue

            # concatenate cumulative sentences when the length near maximum
            if count_tokens + n > max_token:
                chunks.append(
                    ((period_type + " ").join(to_merge) + period_type))
                to_merge.clear()
                count_tokens = 0

            to_merge.append(s)
            count_tokens += n + 1

        # concatenate remaining sentences
        if count_tokens:
            chunks.append(((period_type + " ").join(to_merge) + period_type))

        return chunks

    def embed(self,
              df: pd.DataFrame,
              max_token: int = 500,
              period_type: str = "."):
        """ Embedding the DataFrame

        :param df: a DataFrame contains Page No. and Contents
        :param max_token: the maximum number of token of a model for embedding
        :param period_type: "." for English text and "。" for Chinese text
        :return: a DataFrame contains text: str within the max_token and embeddings: np.ndarray
        """

        print("Generating embeddings...")

        df["num_tokens"] = df["Contents"].apply(
            lambda x: len(self.tokenizer.encode(x)))

        to_token = list()
        for row in df.iterrows():

            # skip no content
            if row[1]["Contents"] is None:
                continue

            # split long text
            if row[1]["num_tokens"] > max_token:
                to_token += self.break_long_text(row[1]["Contents"],
                                                 max_token=max_token,
                                                 period_type=period_type)

            # directly add short text
            else:
                to_token.append(row[1]["Contents"])

        embed_df = pd.DataFrame(to_token, columns=["text"])

        num_rows = len(embed_df)
        embed_df.insert(1, "embeddings", None)
        embed_df.insert(2, "num_tokens", None)

        # generate embeddings
        for i in range(0, num_rows, 60):
            print(
                f"Generating embeddings from rows {i+1} to {min(i+60, num_rows)}..."
            )
            embed_df["embeddings"][i:i + 60] = embed_df["text"][
                i:i + 60].apply(lambda x: openai.Embedding.create(
                    input=x, engine='text-embedding-ada-002')['data'][0][
                        'embedding'])
            embed_df["num_tokens"][i:i +
                                   60] = embed_df["text"][i:i + 60].apply(
                                       lambda x: len(self.tokenizer.encode(x)))

            # the limit of openai.Embedding.create is 60/min
            if i + 60 < num_rows:
                time.sleep(60)

        print("Finish generating!\n")

        return embed_df
