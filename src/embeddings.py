from dotenv import load_dotenv
from openai import OpenAI
import os
import tiktoken

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
TOKEN_LIMIT = os.getenv('EMBEDDING_TOKEN_LIMIT')
ENCODER = os.getenv('ENCODER_MODEL_NAME')

class Embeddings:

    def __init__(self, text, chunk_size: int=None, chunk_overlap: int=0):
        """
        Initializes the Embeddings class with the text to be embedded and the
        chunk size to be used for splitting the text into smaller chunks.

        Args:
            text (str): The text to be embedded.
            chunk_size (int): The size of the chunks to be used for splitting
                the text. If `None`, the text will be split into chunks based
                on the token limit of the embedding model. If the chunk size is
                greater than the token limit of tyhe embedding model, a
                `ValueError` will be raised.
            chunk_overlap (int): The overlap between the chunks. Defaults to 0.        
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.text = text
        if chunk_size is None:
            chunk_size = int(TOKEN_LIMIT)
        elif chunk_size > int(TOKEN_LIMIT): 
            raise ValueError(
                f"Chunk size must be less than or equal to {TOKEN_LIMIT}"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _chunkenize_text(self) -> list[str]:
        """Splits the text into chunks according to chunk_size."""
        encoding = tiktoken.get_encoding(ENCODER)
        tokens = encoding.encode(self.text)
    
        if len(tokens) <= self.chunk_size:
            return [self.text]
        else:
            chunks = []
            start = 0
            end = self.chunk_size

            while end < len(tokens):

                chunk_tokens = tokens[start:end]
                text = encoding.decode(chunk_tokens)
                start += self.chunk_size - self.chunk_overlap
                end += self.chunk_size - self.chunk_overlap
                chunks.append(text)

            # Include the last chunk
            chunk_tokens = tokens[start:]
            text = encoding.decode(chunk_tokens)
            chunks.append(text)

        return chunks


    def get_embeddings(self) -> list[tuple]:
        """
        Returns vector embeddings for the given text breaking the text into
        chunks.

        Returns:
            list[tuple]: A list of tuples where each tuple contains the
                embeddings for a chunk of the text. Each tuple is in the form
                (text, embedding) where `embedding` is a list of floats.
        """
        e = Embeddings(self.text)
        self.chunks = e._chunkenize_text()
        embeddings = []

        for chunk in self.chunks:
            response = e.client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=chunk
            )

            embeddings.append((chunk, response.data[0].embedding))

        return embeddings