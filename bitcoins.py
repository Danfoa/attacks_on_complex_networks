from dgl.data.bitcoinotc import BitcoinOTC
from dgl import DGLGraph
import numpy as np
import pandas as pd
from pytablewriter import RstGridTableWriter


def get_bitcoin_graph():
    writer = RstGridTableWriter()
    bitcoinOTC = BitcoinOTC()
    extract_graph = lambda g: g if isinstance(g, DGLGraph) else g[0]

    graphs = bitcoinOTC.graphs
    num_nodes = []
    num_edges = []
    for i in range(len(graphs)):
        g = extract_graph(graphs[i])
        num_nodes.append(g.number_of_nodes())
        num_edges.append(g.number_of_edges())


    gg = extract_graph(graphs[0])
    dd = {
        "Datset Name": "BitcoinOTC",
        "Usage": bitcoinOTC,
        "# of graphs": len(graphs),
        "Avg. # of nodes": np.mean(num_nodes),
        "Avg. # of edges": np.mean(num_edges),
        "Node field": ', '.join(list(gg.ndata.keys())),
        "Edge field": ', '.join(list(gg.edata.keys())),
        # "Graph field": ', '.join(ds[0][0].gdata.keys()) if hasattr(ds[0][0], "gdata") else "",
        "Temporal": hasattr(graphs, "is_temporal")
    }

    print(dd.keys())
    df = pd.DataFrame([dd])
    df = df.reindex(columns=dd.keys())
    writer.from_dataframe(df)

    writer.write_table()

    G = DGLGraph.to_networkx(graphs[0])
    return G