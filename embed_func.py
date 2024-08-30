import langchain  #LLM Library
import chromadb   #Vector storage

from langchain_openai import OpenAIEmbeddings
from config import PROXY_API

def get_embedding_func():
    embeddings = OpenAIEmbeddings(
        openai_api_key=PROXY_API,
        base_url="https://api.proxyapi.ru/openai/v1",
        show_progress_bar=True,
        chunk_size=3
    )
    return embeddings