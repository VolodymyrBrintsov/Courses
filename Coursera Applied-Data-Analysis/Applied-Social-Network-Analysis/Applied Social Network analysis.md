**Applied social network analysis in Python**

Week 1:

Network is a set of nodes with interconnections(edges)

1. Python representation:

- Import networkx as nx
- G = nx.Graph() # Graph / G = nx.DiGraph()
- G.add\_edge(&#39;a&#39;, &#39;b&#39;) create and edge

1. Some edges carry more weight than others
2. Some network carry information about friendship and antoganist based on conflict or disagreement
3. Edges can carry relation
4. Multigraps: A network where multiple edges can connect the same nodes (parallel graphs, nx.MultiGraphs())
5. G.edges() – get list of all edges

G.edges(data=True) – list of all edges with attributes

1. G.edge[&#39;A&#39;][&#39;B&#39;] – dictrionarry of attributes of edge A and B
2. G.add\_node(&#39;Name&#39;, role=&#39;Name of node&#39;s role&#39;)
3. G.nodes() – list of nodes

G.nodes(data=True) – list of all nodes with attributes

1. Bipartite graphs: a graph whose nodes can be split into two sets L and R and every egde connects and node L and node in R
2. From network.algorithms import bipartite:

B = nx.Graph()

B.add\_nodes\_from([list of nodes], bipartite=0) label one set of nodes 0

B.add\_nodes\_from([list of nodes], bipartite=1) label other set of nodes 1

B.add\_edges\_from([(&#39;A&#39;, 1)…) create edges

1. Bipartite.projected\_graph(B, X)

Week 2:

1. Tradic closure: The tendency for people who share connections in a social network to become connected
2. Local Clustering Coefficient:

- Find the degree of and Node
- Find pairs of nodes ((degree\*(degree-1))/2)
- Find pairs of nodes that are connected and divide by pairs of nodes

1. nx.clustering(graph, &#39;node&#39;)
2. Clustering coeff measures the degree to wich nodes in a network tend to cluster or form triangles
3. Local Clustering Coeff fraction of pairs the nodes friends that are friends with each other
4. Global Clustering Coeff:

- Average local clustering – nx.average\_clustering()
- Transitivity – ration of triangles and numbers of &#39;open triads&#39; nx.transitivity

1. Breadth-first search – a systematic and efficient procedure for computing distance from node to all other nodes in a large network by discovering nodes in layers(nx.bfs\_tree(prag, &#39;node&#39;)
2. Average distance between pairs of nodes nx.average\_shortest\_path\_lenght
3. Maximum distance between any pairs of nodes nx.diameter()
4. Eccentricity of node is the largest distance between n and all other nodes nx.eccentricity()
5. Radius of a graph is the minimum eccentricity nx.radius()
6. Periphery is a set of nodes that have eccentricity equal to a diameter
7. The center of graph is the set of nodes that have eccentricity equal to radius nx.center()
8. An undirected graph is connected if for every pair nodes there is a path between them
9. Nx.node\_connected\_component(graph, &#39;node&#39;) returns nodes that belongs to node
10. A directed graph is strongly connected of for every pair nodes, there is a directed path from one node to other and vice versa nx.is\_strongly\_connected
11. Network robustness the ability of a network to maintain its general structural properties when its faces failers or attacks
12. Nx.node\_connectivity() returns the number of nodes that are connected
13. Nx.minimum\_node\_cut() return the node by deleting which graph become disconnected
14. Nx.edge\_connectivity() returns the number of edges that are connected
15. Nx.minimum\_edge\_cut() return edges by deleting which graph become disconnected
16. Nx.weakly\_connected\_components() return nodes weakly connected components

Week 3:

1. Important nodes have many connections nx.degree\_centrality()/ nx.in\_degree\_centrality()/nx.out\_degree\_centrality()
2. Closness centrality – important nodes are close to other nodes nx.closeness\_centrality()
3. In graph theory, betweenness centrality is a measure of centrality in a graph based on shortest paths. For every pair of vertices in a connected graph, there exists at least one shortest path between the vertices such that either the number of edges that the path passes through (for unweighted graphs) or the sum of the weights of the edges (for weighted graphs) is minimized. The betweenness centrality for each vertex is the number of these shortest paths that pass through the vertex.
4. Nx.betwenness\_centrality(graph, normalized=, endpoints=true/false)
5. Normalization pf betweenness centrality – devide by numbers of pairs of node
6. Approximation – approximate computation by taking subsets of nodes
7. Subsets – we can define subsets of source and target nodes to compute betweenness centrality
8. Edge betweenness centrality – we can apply the same framework to find important edges instead of nodes
9. Basic PageRank(nx.pagerank(graph, alpha=)):

- All Nodes start with PageRank of 1/n(num of nodes)
- Perform the basic PageRank Update rule - each node gives an equal share of its current pagerank to all the nodes it links to. The new pagerank of each node is the sum of all pagerank it received from other nodes

1. Hubs and Authorities, given a query to a search engine:

- Root: set of higly relevant web pages – potential authorities
- Find all pages that link to a page in root – potential hubs
- Base: root nodes and any node that links to a node in a root

1. Hits algorithm hx.hits(graph):

- Assign each node an authority and hub score of 1
- Apply the Authority update rule – each node&#39;s authority score is the sum of hub scores of each node that points to it
- Apply the hub update rule – each node&#39;s hub score is the sum of authority scores of each node that it points to
- Normalize authority score
- Repeat k time

Week 4:

1. Degree Distribution – is the probability distribution of the degrees over the entire network
2. Nx.Barbasi\_albert\_graph(n, m) – returns a network with n nodes. Each new node attaches to m existing nodes according to the preferential attachment model
3. Small World Model – start with a ring of n nodes, where each node is connected to its k nearest neighbors\
4. Nx.watts\_strogatz\_graph(n, k, p) – return a small world network with n nodes, starting with a ring lattice with each node connected to its k nearest neighbors an rewriting probability p
5. Link prediction problem – Given a network, predict which edges will be formed in the future
6. Basic measures for link prediction:

- Number of common neighbors nx.common\_neighbors(Graph)
- Jaccard Coefficient – number of common neighbors normalized by the total number of neighbors nx.jaccard\_coefficient(Graph)
- Resource Allocation Index – fraction of resource that a node can send to another through their common neighbors nx.ressource\_allocation\_index(G)
- Adam-Adar index – similar to resource allocation index but with log in the denominator nx.adamic\_adar\_index(G)
- Preferential attachments – nodes with high degree get more neighbors nx.preferential\_attachment(g)
- Common Neibor Soundarajah-Hopcroft Score – Number of common neighbors but with bonus for neighbors in same community nx.cn\_soundarajan\_hopcroft(G)