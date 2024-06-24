from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


from langchain_openai import OpenAI
from langchain_community.llms.ollama import Ollama

import pandas as pd
import numpy as np
import os
from ollama import Client
import json

import networkx as nx
import seaborn as sns
import random
from pyvis.network import Network


def load_doc(path):
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(documents)


def create_chunks(path, replace_newlines=False):
    document = load_doc(path)
    chunks = split_docs(document)
    if replace_newlines == True:
        for i in range(len(chunks)):
            chunks[i].page_content = chunks[i].page_content.replace("\n","")
        return chunks
    
    return chunks

def graph_prompt(input: str, llm: Client, metadata = {}):
    system_prompt = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    client = Client()
    user_prompt = f"Context to use: {input} \n\n Put output here:"
    #response = llm.generate([system_prompt,user_prompt])
    response = client.generate(model="llama3", prompt=user_prompt, system=system_prompt)
    #response_text = re
    #print(response.keys())
    #print(response['response'])
    #print(type(response))
    #result = json.loads(response)
    #return [dict(item, **metadata) for item in response]
   
    response = response['response']
    #print(response)
    if "[" not in response and "]" not in response:
        print("Error generating response, trying again with input:", input)
        graph_prompt(input, llm)
    x = response.index("[")
    x1 = response.index("]")
    response = response[x:x1+1]

    
    #result = response['response']
    try:
        response = json.loads(response)
    except:
        print("Error in json parsing")
        print(response)
    #f = open("response.txt", "w")
    #f.write(str(response))
   # f.close()
    return [dict(item, **metadata) for item in response]



def contextual_proximity(df: pd.DataFrame):


    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    #print("\n\n\n",dfg2)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    #print("\n\n\n",dfg2)

    dfg2["edge"] = "contextual proximity"
    return dfg2
    
def colors_to_community(communities):
    palette = sns.color_palette("hls", len(communities)).as_hex()
    random.shuffle(palette)
    rows = []
    group = 0
    for community in communities:
        color = palette.pop()
        group += 1
        for node in community:
            rows.append({"node": node, "color": color, "group": group})

    df_colors = pd.DataFrame(rows)
    return df_colors



path = "./PDFs/"
print("Creating chunks from PDFs...")
docs = create_chunks(path, replace_newlines=True)
print("Chunks created")

num_chunks = len(docs)

chunk_id = 0
docs_text = []

df = pd.DataFrame(columns = ["chunk_text", "chunk_source","chunk_id"])

for d in docs:

    df = df._append({"chunk_text":d.page_content, "chunk_source":d.metadata["page"], "chunk_id":str(chunk_id)}, ignore_index=True)
    chunk_id += 1


#print(df.head())
for doc in docs:
    docs_text.append(doc.page_content)

doc1 = docs_text[0]
#for d in docs:
    #print(d.metadata["page"])

model = "llama3"
llm = Ollama(model=model, temperature = 0, top_p = 0.6)



responses = []
print("Number of chunks to create graph from:", len(docs_text))
for i in range(len(docs_text)): 
    response = graph_prompt(docs_text[i], llm)
    
    print(f"Chunk {i} completed")
    #print(response)
    responses.append(response)
    #if i == 4:#REMOVE LATERREMOVE LATERREMOVE LATERREMOVE LATERREMOVE LATERREMOVE LATER
    #    break
    #break#NEED TO REMOVE THIS BREAK LATER




nodes_df = pd.DataFrame(columns = ["node_1", "node_2", "edge", "chunk_id"])

for response in responses:
    #print("Response:",response[0])
    
    
    count = 0
    for item in response:

        item["chunk_id"] = str(count)
        #print(list(item.values()))
        #print(r)
        #print(len(list(item.values())), len(nodes_df.columns))
        l = list(item.values())
        #df.loc[df.index] = [1,2,3,4]
        nodes_df = nodes_df._append(item, ignore_index=True)
        #df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
        count += 1
    
for i in range(3):
    print(nodes_df.iloc[i])
    print("\n\n")

#print(nodes_df.head())  

nodes_df2 = contextual_proximity(nodes_df)
#print(nodes_df2)

nodes_df = pd.concat([nodes_df, nodes_df2], axis=0)
nodes_df = (
    nodes_df.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)

#print(nodes_df)


nodes = pd.concat([nodes_df["node_1"], nodes_df["node_2"]], axis = 0).unique()
print("\n\n\n\n")
#print(nodes_df.columns)

G = nx.Graph()

nodes_df.to_pickle("nodes_df.pkl")


for node in nodes:
    G.add_node(str(node))

#for x in nodes_df.iterrows():
    #print(x)


#print(nodes_df.columns)
for index, row in nodes_df.iterrows():
    #print(row)
    
    #print(row["edge"])
    G.add_edge(str(row["node_1"]), str(row["node_2"]), title = row["edge"], weight = row["count"]/4)

communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of communities:",len(communities))
print(communities)

file = open("communities.txt", "w")
file.write(str(communities))
file.close()


print("Communities are:",type(communities))
palette = sns.color_palette("hls", len(communities))


colors = colors_to_community(communities)

for index, row in colors.iterrows():
    G.nodes[row["node"]]["color"] = row["color"]
    G.nodes[row["node"]]["group"] = row["group"]   
    G.nodes[row["node"]]["size"] = G.degree[row["node"]]


graph_path = "./graphs/index.html"

net = Network(notebook = False, cdn_resources= "remote", height = "900px", width = "100%", select_menu=True, filter_menu=False)


net.from_nx(G)
net.force_atlas_2based(central_gravity=0.015, gravity = -31)
net.show_buttons(filter_=['physics'])
net.show(graph_path, notebook = False)