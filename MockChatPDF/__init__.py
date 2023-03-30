"""Author: Junxiao Zhao"""

from . import embedpdf
from . import chatpdf

EmbedPDF = embedpdf.EmbedPDF
"""This is a class for converting PDF to a DataFrame with embeddings of the text

:param file_path: the path of the PDF
"""

ChatPDF = chatpdf.ChatPDF
"""This model take a DataFrame as input and could answer question related to it with the help of ChatGPT
        
:param embedded_df: a DataFrame with columns [text, embeddings, num_tokens]
"""
