import os
from getpass import getpass
from pathlib import Path

import chainlit as cl
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter your OpenAI API key: "
)

SYSTEM_PROMPT = """You are a virtual assistant for the Central Bank of Houston.
Your goal is to assist users in using the bank app.
You must use the relevant documentation given to you to answer user queries.
You can only answer questions about the bank app."""

# RAG retriever tool
loader = PyPDFLoader(Path(__file__).parent / "Online_Banking_User_Guide.pdf")
pages = loader.load_and_split()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(pages)
print("Indexing documents...")
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
print("done!")
retriever = vector.as_retriever()


@cl.on_chat_start
async def on_chat_start():
    # LLM client
    client = openai.OpenAI(
        # base_url="",
        # default_headers={
        #     "X-APORIA-API-KEY": "",
        # },
    )
    cl.user_session.set("client", client)

    # Chat history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("chat_history", chat_history)

    cl.user_session.set("retriever", retriever)

    welcome_message = (
        "Welcome to the Central Bank of Houston App support center."
        " How may I assist you?"
    )

    chat_history.append({"role": "assistant", "content": welcome_message})
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    client: openai.OpenAI = cl.user_session.get("client")
    chat_history: list = cl.user_session.get("chat_history")
    retriever: VectorStoreRetriever = cl.user_session.get("retriever")

    try:
        chat_history.append({"role": "user", "content": message.content})

        # Retrieve documents relevant to the user's message.
        # Then, add the relevant documents to the chat history.
        relevant_documents = retriever.get_relevant_documents(message.content)
        if len(relevant_documents) > 0:
            relevant_info = "\n\n".join(
                map(lambda doc: doc.page_content, relevant_documents)
            )
            chat_history.append(
                {"role": "system", "content": f"Relevant Documents: \n{relevant_info}"}
            )

        # Generate a response using our LLM client.
        response = client.chat.completions.create(model="gpt-4o", messages=chat_history)
        reply = response.choices[0].message.content

        # Append the assistant's reply to the chat history
        chat_history.append({"role": "assistant", "content": reply})
        cl.user_session.set("chat_history", chat_history)

        # Send the assistant's reply back to the user
        await cl.Message(content=reply if reply is not None else "").send()

    except Exception as e:
        # Handle any exceptions that occur during the API request
        await cl.Message(content=f"An error occurred: {str(e)}").send()


@cl.on_stop
async def on_stop():
    # Clear the chat history when the session ends
    cl.user_session.set("chat_history", [])
