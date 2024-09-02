from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from config import PROXY_API
from query_data import query_rag

# История диалога: {history}

# PROMT_STILE = '''
# Отредактируй content сообщения с помощью тегов ниже для визуального понимания главной мысли сообщения в телеграме, используя:
# <b>жирный текст</b>
# <i>Курсив</i>
# <u>подчеркнутый</u>
# 
# Сообщение: 
# {message}
# '''
PROMT_STILE = '''
Сохраняя и не меняя содержимое текста - перепиши его в нужный формат:
-Ключевая информация выделена только с помощью <b>ключевой текст</b>
-Добавлены абзацы

Исходный текст ответа:
{message}
'''

#Работает форматирование текста
async def style_telegram(message: str):
    prompt_template = ChatPromptTemplate.from_template(PROMT_STILE)
    prompt = prompt_template.format(message=message)
    print(prompt)

    model = ChatOpenAI(
        model_name='gpt-4o-mini',
        openai_api_key=PROXY_API,
        base_url="https://api.proxyapi.ru/openai/v1"
    )
    response_text = model.invoke(prompt)
    print(f"[info coroutine] {response_text.content}")
    return str(response_text.content)



