# Переменные для ввода при запуске контейнера: OPENAI_API_KEY, CH_HOST, CH_PORT, CH_USER, CH_PASSWORD
import os
import re
import logging
from langchain.agents import AgentExecutor,  initialize_agent # Tool,
from langchain_core.tools import Tool
from langchain.docstore.document import Document
from langchain.agents.agent_types import AgentType
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from openai import OpenAI
import clickhouse_connect

# Настройка логирования
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.info("START_LOGGING")

# Конфигурация безопасности
os.environ['AGENT_MODE'] = 'STRICT'  # Запрет управления агентом

# Получение ключа API от пользователя и установка его как переменной окружения
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')  

client = ChatOpenAI(model="gpt-4", temperature=0)

# подключение к БД параметры подключения - в команде запуска контейнера
def connect_to_base(host, port, user, password):
    '''
    Подключение к БД
    '''
    try:
        client = clickhouse_connect.get_client(host= host, port= port, username= user, password= password)
        logger.debug("Connection to the database is successful")
        return client
    except:
        logger.error(f"Error loading to connect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
    
client_ch = connect_to_base(os.environ.get('CH_HOST'), os.environ.get('CH_PORT'), os.environ.get('CH_USER'), os.environ.get('CH_PASSWORD'))  
    
    
# Инструмент 1: Безопасный ClickHouse запрос
def safe_clickhouse_query(query: str) -> str:
    """Выполняет SQL-запросы только для чтения с валидацией"""
    # Защита от инъекций
    forbidden_keywords = ['insert', 'update', 'delete', 'drop', 'alter', 'create', 'grant']
    if any(re.search(rf'\b{kw}\b', query.lower()) for kw in forbidden_keywords):
        return "Ошибка: Запрещенная операция"
    
    # Только SELECT/SHOW/DESCRIBE
    if not re.match(r'^\s*(select|show|describe|with|explain)', query, re.IGNORECASE):
        return "Ошибка: Разрешены только запросы чтения"
    
    try:
        result = client_ch.query_df(query)
        return result
    except Exception as e:
        return f"Ошибка запроса: {str(e)}"    
    

# Разбиение документа на чанки
def get_chunks(splitter, text):
    chunks = []
    for chunk in splitter.split_text(text):
        # Если chunk - объект Document, используем его page_content
        if hasattr(chunk, 'page_content'):
            page_content = chunk.page_content
            metadata = getattr(chunk, 'metadata', {}).copy()  # Копируем существующие метаданные, если они есть
            metadata.update({"meta": "data"})  # Обновляем их
            chunks.append(Document(page_content=page_content, metadata=metadata))
        else:
            # Если chunk - строка (на всякий случай)
            chunks.append(Document(page_content=chunk, metadata={"meta": "data"}))
    return chunks


# Инструмент 2: RAG для схемы базы данных
def setup_rag_agent():
    """Инициализация векторной базы знаний о схеме"""
    file_md = open('db_schema_docs.md', encoding='utf-8').read()
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),
                                                                   ("##", "Header 2"),],
                                              strip_headers = False)
    chunks = get_chunks(text_splitter, file_md)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

vector_db = setup_rag_agent()

def schema_retriever(query: str) -> str:
    """Поиск информации о структуре БД"""
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])


# Инструмент 3: Python для сложных вычислений
python_repl = PythonREPL()

# Инициализация инструментов
tools = [
    Tool(
        name="ClickHouse_Query",
        func=safe_clickhouse_query,
        description=(
            "Выполнение SQL-запросов к ClickHouse. Только SELECT/SHOW/DESCRIBE. "
            "Вход: валидный SQL-запрос. Выход: результат таблицы."
        )
    ),
    Tool(
        name="Database_Schema",
        func=schema_retriever,
        description=(
            "Поиск информации о структуре базы данных. "
            "Используй для уточнения имен таблиц и столбцов. "
            "Вход: естественный язык."
        )
    ),
    Tool(
        name="Python_REPL",
        func=python_repl.run,
        description=(
            "Выполнение Python кода для сложных вычислений. "
            "Используй только когда невозможно решить через SQL. "
            "Вход: валидный Python код."
        )
    )
]

# Системный промпт с ограничениями
system_prompt = f"""
Ты senior data analyst. Правила:
1. Только чтение данных (SELECT/SHOW/DESCRIBE)
2. Запрещены DDL/DML операции
3. Для работы с БД используй инструменты
4. Все ответы на русском
5. Формат вывода: сформированный SQL-запрос и таблица Markdown

Доступные базы:
{vector_db}
"""

# Инициализация агента
agent = initialize_agent(
    tools=tools,
    llm = client,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs={
        'system_message': system_prompt,
        'extra_prompt_messages': [
            "Важно: Все SQL-запросы проверяй через Database_Schema!",
            "Ограничение: Максимум 5 строк в Python_REPL"
        ]
    },
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":

    while True:
        query = input('Вопрос пользователя: ')
        # выход из цикла, если пользователь ввел: 'стоп'
        if query == 'стоп': break
        # ответ от OpenAI
        result = agent.invoke({"input": query})

        print(f"\nРезультат:\n{result['output']}")
