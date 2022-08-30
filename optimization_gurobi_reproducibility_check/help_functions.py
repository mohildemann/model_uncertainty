import matplotlib.pyplot as plt
import networkx as nx
import copy


def draw(shape, size_tuple=(10, 10)):
    "Function to visualize shapefile with its corresponding IDs"
    shape['coords'] = shape['geometry'].apply(lambda x: x.representative_point().coords[:])
    shape['coords'] = [coords[0] for coords in shape['coords']]
    shape.plot(figsize=size_tuple, edgecolor="purple", facecolor="pink")
    for idx, row in shape.iterrows():
        plt.annotate(text=row['id'], xy=row['coords'],
                     horizontalalignment='center')


def queen(shape):
    "Function that finds queen contiguity. Returns a list containing the adjacency (i,j)"

    shape['id'] = shape.index

    # shape.plot(figsize=(5,5), edgecolor="purple", facecolor="None")

    # add NEIGHBORS column
    shape["NEIGHBORS"] = None

    for index, country in shape.iterrows():
        # get 'not disjoint' countries
        neighbors = shape[~shape.geometry.disjoint(country.geometry)].id.tolist()

        # remove own name of the country from the list
        neighbors = [name for name in neighbors if country.id != name]

        # add names of neighbors as NEIGHBORS value
        shape.at[index, "NEIGHBORS"] = neighbors

    edges2 = []
    # sink = shape['id'].iloc[-1]
    for i in range(len(shape['id'])):
        for j in range(len(shape['NEIGHBORS'][i])):
            # if i!=sink:
            edges2.append((shape['id'][i], shape['NEIGHBORS'][i][j]))
    return edges2


def rook1(shape):
    "Function that finds rook contiguity. Returns a list containing the adjacency (i,j) "

    shape['id'] = shape.index

    # shape.plot(figsize=(5,5), edgecolor="purple", facecolor="None")

    # add NEIGHBORS column
    shape["NEIGHBORS"] = None

    for index, country in shape.iterrows():
        # get 'not disjoint' countries
        neighbors = shape[shape.geometry.exterior.overlaps(country.geometry.exterior)].id.tolist()

        # remove own name of the country from the list
        neighbors = [name for name in neighbors if country.id != name]

        # add names of neighbors as NEIGHBORS value
        shape.at[index, "NEIGHBORS"] = neighbors

    # print(shape['NEIGHBORS'])
    edges2 = []
    # sink = shape['id'].iloc[-1]
    for i in list(shape['id']):
        for j in range(len(shape['NEIGHBORS'][i])):
            # if i!=sink:
            edges2.append((shape['id'][i], shape['NEIGHBORS'][i][j]))
    return edges2

def geod_to_graph1(geodataframe, adj='rook'):
    if adj == 'rook':
        edges2 = rook1(geodataframe)
    else:
        edges2 = queen(geodataframe)
    G = nx.DiGraph()
    G.add_edges_from(edges2)
    return G

def contiguous_patch_with_area_limit(node, gdf, var_area, T):

    "This function returns a set of contiguos spatial units that comprise the largest patch allowed by the threshold T."
    f = gdf.iloc[node][var_area]
    #get the direct neighbors of an unit and save in Ni as possible candidate unit list
    Ni = gdf.iloc[node]["NEIGHBORS"]
    # phi is the final set of the contiguous area
    phi = []

    while f < T:
        phi.append(node)
        # k = dataframe rows of direct neighbors to the unit
        k = gdf[gdf['id'].isin(Ni)]
        # sort by area (ascending)
        k = k.sort_values([var_area])

        # check whether the area of the unit plus the first (largest area) neighbor exceeds the threshold
        # here, the node is changed to the smallest area unit
        if k.iloc[0][var_area] + f < T:
            # get id of unit with smallest area, set as new node and remove from NI
            node = k.iloc[0]['id']
            # remove node from possible candidate list
            Ni.remove(node)
            # add all direct neighbors of the selected unit to the possible candidate list Ni
            Ni = list(set(Ni + gdf.iloc[node]["NEIGHBORS"]))
            Naux = copy.deepcopy(Ni)
            # Here, all units are removed from the possible candidate list Ni if they are already in phi
            for i in Naux:
                if i in phi:
                    try:
                        Ni.remove(i)
                    except:
                        pass
        f = f + k.iloc[0][var_area]
    return phi

def sink_ord(node,gdf):
    "This function finds the set of units of i that have a lower index than i"
    Ni = gdf.iloc[node]['NEIGHBORS']
    sink_order = [a  for a in Ni if a < node]
    return sink_order