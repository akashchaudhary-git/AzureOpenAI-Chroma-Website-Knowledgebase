from typing import Optional

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import os
import openai
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from loguru import logger



load_dotenv()

def extract_urls_from_sitemap(sitemap):
    """
    Extract all Urls from a sitemap XML string.

    Args:
        sitemap_string (str): The sitemap XML string.

    Returns:
        A list of Urls extracted form the sitemap.
    """

    # Parse the XML from the string
    root = ET.fromstring(sitemap)

    # Define the namespace for the sitemap XML
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Find all <loc> elements under the <url> elements
    urls = [
        url.find("ns:loc",namespace).text for url in root.findall("ns:url", namespace)
    ]

    # Return the list of Urls
    return urls



class KnowledgeBase:
    def __init__(
            self,
            sitemap_url:str,
            chunk_size:int,
            chunk_overlap:int,
            pattern: Optional[str] = None):
        """
            Build a knowledge base from a sitemap.
            args:
                sitemap_url (str): The Url of the sitemap.
                chunk_size (int): The size of the chunks to split the documents into.
                chunk_overlap (int): The overlap between chunks.
                pattern (str): A pattern to filter Urls with.
        """

        logger.info("Building the knowledge base ...")

        logger.info("Loading sitemap from {sitemap_url} ...", sitemap_url=sitemap_url)
        sitemap = requests.get(sitemap_url).text
        urls = extract_urls_from_sitemap(sitemap)

        if pattern:
            logger.info("Filtering Urls with pattern {pattern} ...", pattern=pattern)
            urls = [x for x in urls if pattern in x]
        logger.info("{n} Urls extracted", n=len(urls))


        logger.info("Loading Urls content ...")
        loader = UnstructuredURLLoader(urls)
        data = loader.load()


        logger.info("Splitting documents in chunks ...")
        doc_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = doc_splitter.split_documents(data)
        logger.info("{n} chunks created", n=len(docs))

        logger.info("Building the vector database ...")


        # Configure Azure OpenAI Service API
        openai.api_type = "azure"
        openai.api_version = "2022-12-01"
        openai.api_base = os.getenv('OPEN_AI_BASE')
        openai.api_key = os.getenv("OPEN_AI_KEY")

        embeddings = OpenAIEmbeddings(
            deployment= "ac-text-embedding-ada-002", 
            chunk_size = 1
            )
        docsearch = Chroma.from_documents(docs,embeddings)

        logger.info("Building the retrieval chain ...")
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            AzureChatOpenAI(deployment_name="ac-gpt-35-turbo",openai_api_base=os.getenv("OPEN_AI_BASE"),openai_api_key=os.getenv("OPEN_AI_KEY"),openai_api_type="azure",openai_api_version="2023-03-15-preview",model_name="gpt-35-turbo"), 
            chain_type="stuff", 
            retriever=docsearch.as_retriever(),
            )
        
        logger.info("Knowledge base created!")


    def ask(self, query:str):
        return self.chain({"question":query}, return_only_outputs=True)
    


if __name__ == "__main__":
    # Build Knowledge base
    kb = KnowledgeBase(
        sitemap_url="https://nextjs.org/sitemap.xml",
        pattern="docs/app/api-reference/create-next-app",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Ask a question

    res = kb.ask("Why use Create Next App?")
    print(res['answer'])
    print(res['sources'])


