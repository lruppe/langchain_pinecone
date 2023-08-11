import os

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone


if __name__ == "__main__":
    print("hello vectorstore")
    load_dotenv()
    env_variables = dict(os.environ)

    pinecone.init(
        api_key=env_variables["PINECONE_API_KEY"],
        environment="asia-southeast1-gcp-free",
    )

    path = os.path.join(os.getcwd(), "mediumblogs", "mediumblogs.txt")

    loader = TextLoader(path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=env_variables["OPENAI_API_KEY"])
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector DB? Explain it in the words of Shakespeare."
    result = qa({"query": query})
    print(result)
