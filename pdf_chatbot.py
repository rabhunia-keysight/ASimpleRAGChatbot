from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load and split PDF
loader = UnstructuredPDFLoader("your_file.pdf")  # Replace with your actual PDF file
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create embeddings and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# Initialize LLM and prompt
llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the context below:\n\n{context}"),
    ("human", "{question}")
])
output_parser = StrOutputParser()

# Build the chain
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = (
    {"context": retriever | format_docs, "question": lambda x: x["question"]}
    | prompt
    | llm
    | output_parser
)

# Run chatbot
if __name__ == "__main__":
    print("PDF Chatbot is ready. Type your questions or 'exit' to quit.")
    while True:
        query = input("\n\nAsk a question: ")
        if query.lower() == "exit":
            break
        result = chain.invoke({"question": query})
        print(f"\nAnswer: {result}")

