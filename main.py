import make_graphv2 as mg
from ollama import Client


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
    x = client.generate(model="llama3", prompt = f"Here is the community, please summarize it: {community}", system="You are analyzing a knowledge graph with nodes grouped into communities. Please summarize the given community in 3 or more sentences.")   
    print(x['response'])
    print("\n\n")
    community_summaries.append(x['response'])

