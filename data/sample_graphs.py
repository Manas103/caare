import networkx as nx

def triangle_graph():
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(0,2)])
    return G

def square_with_diag():
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(0,2)])
    return G
