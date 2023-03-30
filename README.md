# Mock ChatPDF

This package [MockChatPDF](./MockChatPDF/) allows users to extract text from PDF files, and generates answers to related questions using the ChatGPT model. The package utilizes embeddings to calculate the distance between the question and the extracted text, and finds the most relevant context to generate contextually appropriate answers.

Instructions on how to use the package can be found in the [sample.py](./sample.py) file, which provides a sample implementation of the package's functionality.

### Notes
- Use different periods (like "。" for Chinese text and "." for English text) for different text in `MockChatPDF.EmbedPDF.format_text`
- Set the "limit_per_min" in `MockChatPDF.EmbedPDF.embed` according to your plan

### References
> "How to Build an AI That Can Answer Questions about Your Website", OpenAI, https://platform.openai.com/docs/tutorials/web-qa-embeddings. 