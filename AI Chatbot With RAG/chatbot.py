from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Connect to vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def chatbot(question):

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
    Answer the question using the context below.

    If the answer is not available in the context,
    say that you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content