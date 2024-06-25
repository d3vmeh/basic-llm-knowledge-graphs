import make_graphv2 as mg
from ollama import Client
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd

def get_response(query,context1,prompt,model,conversations=""):
    
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
    prompt_template = ChatPromptTemplate.from_template(prompt)
    s = ""
    for c in context1:
        s += c + "\n"
    print(s)
    prompt = prompt_template.format(community_summaries=s, question=query)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc in s]
    formatted_response = f"Response: {response_text}\n"#Sources: {sources}"
    return formatted_response, response_text
#nodes_df = mg.preprocess_text("./PDFs/")
#mg.save_df(nodes_df)

nodes_df = mg.load_df()

print("\ncreating communintites...")
#nodes_df, nodes = mg.get_context(nodes_df)
#G = mg.set_graph_object(nodes_df, nodes)
#g, communities = mg.create_communities(G)

print("Communities created and saved")

#mg.create_graph()
communities = mg.load_communities()

print(nodes_df)

print("===================================================")
print(communities[0])
print("===================================================")








#print(g.nodes)
community_summaries = []
for i, community in enumerate(communities):
    print(f"Community {i+1}: {community}")
    client = Client()
    a = client.generate(model="llama3", prompt = f"Here is the community, please create a title for it: {community}", system="You are analyzing a knowledge graph with nodes connected by relationships and grouped into communities. Please create a title for the given community.")   
    x = client.generate(model="llama3", prompt = f"Here is the community, please summarize it: {community}", system="You are analyzing a knowledge graph with nodes connected by relationships and grouped into communities. Please summarize the given community 2 sentences.")   

    
    #print(x['response'])
    #print("\n\n")
    community_summaries.append(x['response'])



prompt = """
Answer the question only using the community summaries. Here are the community summaries you can use:

Here is the context you can use to help you answer the questions. Keep the answer brief:
{community_summaries}

If you do not know the answer, do not make up an answer, just say you do not know. 

Answer the question based on the above context: {question}
"""


model = Ollama(model="llama3")

while True:
    q = input("Enter a question: ")
    if q.lower() == "exit":
        exit()
    x, response_text = get_response(q,community_summaries,prompt,model)
    print(response_text)
