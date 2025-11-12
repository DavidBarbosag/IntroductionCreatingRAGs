import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; LangChainBot/1.0)"

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("No se encontró la clave OPENAI_API_KEY. Verifica tu archivo .env")

# Inicializar modelos
llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant information from the vector store to answer a question."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject retrieved context into the prompt for better answers."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are a helpful assistant. Use the following context to answer the question:\n\n"
        f"{docs_content}"
    )
    return system_message

agent = create_agent(llm_model, tools=[], middleware=[prompt_with_context])

query = "What is task decomposition?"
print("\n=== Pregunta del usuario ===")
print(query)

print("\n=== Respuesta generada ===")
for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    msg = step["messages"][-1]
    if hasattr(msg, "content"):
        print(msg.content)
