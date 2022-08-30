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