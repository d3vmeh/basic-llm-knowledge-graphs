from knowledge_graph_maker import GraphMaker, Ontology#, #GroqClient
from knowledge_graph_maker import Document as DC
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAI

def load_doc(path):
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(documents)


def create_chunks(path, replace_newlines=False):
    document = load_doc(path)
    chunks = split_docs(document)
    if replace_newlines == True:
        for i in range(len(chunks)):
            chunks[i].page_content = chunks[i].page_content.replace("\n","")
        return chunks
    
    return chunks

ontology = Ontology(
    # labels of the entities to be extracted. Can be a string or an object, like the following.
    labels=[
        {"Person": "Person name without any adjectives, Remember a person may be references by their name or using a pronoun"},
        {"Object": "Do not add the definite article 'the' in the object name"},
        {"Event": "Event event involving multiple people. Do not include qualifiers or verbs like gives, leaves, works etc."},
        "Place",
        "Document",
        "Organisation",
        "Action",
        {"Miscellanous": "Any important concept can not be categorised with any other given label"},
    ],
    # Relationships that are important for your application.
    # These are more like instructions for the LLM to nudge it to focus on specific relationships.
    # There is no guarentee that only these relationships will be extracted, but some models do a good job overall at sticking to these relations.
    relationships=[
        "Relation between any pair of Entities",
        ],
)

#class Document:
    #def __init__(self, text):
        #self.text = text

class LLM:
    def __init__(self, model, temperature, top_p):
        pass

    def generate(self, user_message, system_message):
        pass

path = "./PDFs/"
original_docs = create_chunks(path, replace_newlines=True)
docs = []
for doc1 in original_docs:
    page_content = doc1.page_content

    doc = DC(text = page_content, metadata={"source": "PDF"})
    docs.append(doc)

print(docs[0])

#model = "gpt-3.5-turbo"
#llm = OpenAI(model=model, temperature=0.1, top_p=0.5)


model = "llama3"
llm = Ollama(model=model, temperature=0.1, top_p=0.5)

class test:
    def __init__(self,llm_client, user_message, system_message):
        self.llm_client = llm_client
        self.user_message = user_message
        self.system_message = system_message


    

    def generate(self):

        response = self.llm_client.generate(user_message = self.user_message, system_message = self.system_message)
        return response




#print(llm.generate(["What is the capital of France?"]))
#llm = LLM(model=llm, temperature=0.1, top_p=0.5)

#t = test(llm, ["What is the capital of France?"], ["Answer in two sentences"])
#print(t.generate())


graph_maker = GraphMaker(ontology=ontology, llm_client=llm)

print(type(docs))
graph = graph_maker.from_documents(docs)


print("Number of edges:", len(graph))