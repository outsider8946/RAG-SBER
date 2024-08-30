from langchain_core.documents import Document
from embed_func import get_embedding_func
from langchain_community.vectorstores import Chroma
import argparse
import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
CHROMA_PATH = "chroma"
DATA_PATH = "data"

url = ["https://www.sberbank.com/ru/s_m_business/open-accounts?utm_source=yandex&utm_medium=cpc&utm_campaign=open-accounts_corporate_perform_god_20220100016_rk436516gr2234_context_search_brand_optimisation_rus_yxprrko%7C111959248%7Cgid%7C5458249861%7Cad%7C16234457890_16234457890%7Cph_id%7C52166801428%7Csrc%7Cnone_search%7Cgeo%7CМосква_213%7C&utm_term=сбер+легкий+старт+тарифы&yclid=18330555996781150207",
       "http://government.ru/sanctions_measures/measure/65/",
       "http://government.ru/sanctions_measures/measure/4/",
       "https://www.nalog.gov.ru/rn77/anticrisis2022/#t2",
       "https://мсп.рф/services/knowledge-base/detail/sistema-bystrykh-platezhey-chto-vazhno-znat/",
       "https://мсп.рф/services/knowledge-base/detail/kak-podgotovitsya-k-proverke-chto-nuzhno-znat/",
       "https://мсп.рф/services/knowledge-base/detail/otchetnost-v-nalogovuyu/",
       "https://www.sberbank.com/ru/s_m_business/actions",
       "https://www.sberbank.com/ru/s_m_business/promo/bc_actions",
       "https://www.sberbank.com/ru/s_m_business/bankingservice/sberpay-qr"]

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = PDFload_docs()
    chunks = split_docs(documents)
    add_to_chroma(chunks)

    documents = WEBload_docs(urls=url)
    chunks = split_docs(documents)
    add_to_chroma(chunks)


def PDFload_docs():
    documents_loader = PyPDFDirectoryLoader('data/')
    return documents_loader.load()

def WEBload_docs(urls: list):
    # documents_loader = WebBaseLoader(web_paths=urls, show_progress=True, header_template=headers, verify_ssl=False)
    documents_loader = WebBaseLoader(web_paths=urls, header_template=headers, verify_ssl=False)
    return documents_loader.load()

# Проверить работу на одной папке, WebBaseLoader
# https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/

# chunk_size=800: Это параметр класса RecursiveCharacterTextSplitter,
# который указывает максимальный размер каждого фрагмента текста в символах.
# В этом случае каждый фрагмент будет иметь максимум 800 символов.

# chunk_overlap=80: Это параметр класса RecursiveCharacterTextSplitter,
# который указывает количество символов, которые будут перекрываться между соседними фрагментами текста.
# В этом случае каждый фрагмент будет перекрываться с соседними фрагментами на 80 символов.

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# ValueError: Batch size * exceeds maximum batch size 166
def add_to_chroma(chunks: List[Document]):
    # Загружаем существующую базу данных.
    db = Chroma(
        persist_directory=CHROMA_PATH,  # Путь к директории, где база данных сохраняется на диске
        embedding_function=get_embedding_func(),  # Функция для создания векторных представлений документов
        collection_metadata={"hnsw:space": "cosine"} # Метаданные для настройки пространства поиска -> косинусное расстояние
    )
    # Вычисляем идентификаторы для каждого чанка (части документа).
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Получаем существующие документы из базы данных.
    existing_items = db.get(
        include=[])  # Получаем только идентификаторы документов, так как они всегда включены по умолчанию
    existing_ids = set(existing_items["ids"])  # Преобразуем список идентификаторов в множество для быстрого поиска
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Отбираем только те документы, которых еще нет в базе данных.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        # Обрабатываем документы пакетами по 166 штук для оптимизации производительности.
        batch_size = 166
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i + batch_size]
            batch_ids = new_chunk_ids[i:i + batch_size]
            db.add_documents(batch_chunks,
                             ids=batch_ids)  # Добавляем пакет документов в базу данных с указанием их идентификаторов

        db.persist()  # Сохраняем изменения в базе данных на диск
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # "data/Adobe....pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
    # Если идентификатор страницы совпадает с предыдущим, увеличьте индекс.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        # Вычисление id для чанка
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        # Добавьте его в метаданные страницы.
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
