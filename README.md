# ASimpleRAGChatbot
A simple RAG pipeline using LangChain that works with local PDFs and Ollama

# How to run
Run ollama locally.

```
ollama pull llama3
ollama serve
ollama pull nomic-embed-text
ollama list
ollama run nomic-embed-text
```

Create the python environment and run the chatbot.

```
python3.10 -m venv myvenv
pip install -r requirements.txt
python3.10 pdf_chatbot.py
```
