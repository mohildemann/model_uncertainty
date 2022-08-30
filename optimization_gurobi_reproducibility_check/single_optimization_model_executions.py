import geopandas as geop
from timeit import default_timer as timer
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from help_functions import geod_to_graph1, sink_ord, draw, queen, rook1, contiguous_patch_with_area_limit
import os
import matplotlib.pyplot as plt

e = gp.Env(empty=True)
#e.setParam('WLSACCESSID', '222f6f86-a313-4097-ab9e-1a3b5b8f5171')
#e.setParam('WLSSECRET', 'b76140af-e034-4231-91a4-8ac385ebed52')
e.setParam('USERNAME', r'jhildema@uni-muenster.de')
e.setParam('LICENSEID', 792691)
e.start()

dir = r'input/Subset_LMU_57'
rfile = 'LMU57_all_departures.shp'
column_forsys = "res_depart"
column_landfire = "dep_landfi"
column_landclim = "dep_landcl"

gdf_forsys = geop.read_file(os.path.join(dir, rfile))
gdf_forsys['benefit_forsys']=gdf_forsys[column_forsys]*gdf_forsys['Acres']
gdf_forsys['id'] = gdf_forsys.index

gdf_landfire = geop.read_file(os.path.join(dir, rfile))
gdf_landfire['benefit_landfire']=gdf_landfire[column_landfire]*gdf_landfire['Acres']
gdf_landfire['id'] = gdf_landfire.index

gdf_landclim = geop.read_file(os.path.join(dir, rfile))
gdf_landclim['benefit_landclim']=gdf_landclim[column_landclim]*gdf_landclim['Acres']
gdf_landclim['id'] = gdf_landclim.index

var_obj1 = 'benefit_forsys'
var_obj2 = 'benefit_landfire'
var_obj3 = 'benefit_landclim'

var_area = 'Acres'
id_var = 'id'
T = 100
G = geod_to_graph1(gdf_forsys,adj = 'rook')

#Preprocessing
#Parameters
t1 = timer()

M = [len(contiguous_patch_with_area_limit(i,gdf_forsys,var_area,T)) for i in gdf_forsys[id_var]]
M1 = len(gdf_forsys)
R = [sink_ord(i,gdf_forsys) for i in range(len(gdf_forsys))]
#M = len(gdf_forsys)

#Graph for adjacency
t2 = timer()
print('Massaging time ',t2-t1, ' seconds' )

def shirabi(gdf, T, var_opt, var_area, id_var, objective_function = "meta-model", min_max = "maximize", title = "No title given"):
    t2 = timer()

    # Model set-up
    G = geod_to_graph1(gdf, adj='queen')
    m = None
    m = gp.Model(env=e)
    X = sorted(list(G.nodes()))
    Y = list(G.edges())
    V = sorted(list(G.nodes()))

    # Add variables
    x = m.addVars(X, lb=0, vtype=GRB.BINARY, name='X')
    y = m.addVars(Y, lb=0, vtype=GRB.CONTINUOUS, name='Y')
    v = m.addVars(V, lb=0, vtype=GRB.BINARY, name='V')
    m.update()

    ### Add constraints
    m.addConstr(quicksum(x[i] * gdf.iloc[i][var_area] for i in X) <= T, name='C2')
    m.update()
    m.addConstrs((quicksum((y[i, j] - y[j, i]) for j in list(G[i])) >= x[i] - M[i] * v[i] for i in X), name='C3')
    m.update()
    m.addConstr(quicksum(v[i] for i in V) == 1, name='C4')
    m.update()
    m.addConstrs((quicksum(y[i, j] for j in list(G[i])) <= (M[i] - 1) * x[i] for i in X), name='C5')
    m.update()
    m.addConstrs((v[i] <= x[i] for i in X), name='C6')
    m.update()
    m.addConstrs((x[i] <= 1 for i in X), name='C7')
    m.update()
    # m.addConstrs(((len(R[i]))*v[i]+quicksum(x[j] for j in R[i]) <= (len(R[i])) for i in X), name = 'C10')
    # m.update()
    m.addConstrs((v[i] + x[j] <= 1 for i in X for j in R[i]), name='C11')
    m.update()
    # m.addConstrs((x[i]-v[i] <= quicksum(x[j] for j in list(G[i])) for i in X), name = 'C14')
    # m.update()

    # m.write('wildfire_shirabi.lp')

    t3 = timer()

    print('Model set-up ', t3 - t2, ' seconds')

    ##optimize
    if min_max == "maximize":
        min_max = GRB.MAXIMIZE
    else:
        min_max = GRB.MINIMIZE
    ### Objective
    if objective_function == "meta-model":
        ob_1 = quicksum(x[i] * gdf[var_opt[0]][i] for i in X)
        m.setObjective(ob_1, min_max)
        m.update()
        m.write('wildfire_shirabi.lp')
        # solve
        m.optimize()

        m.setParam(GRB.Param.OutputFlag, 0)

        # Status checking
        status = m.Status

        if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            if m.status == GRB.INFEASIBLE:
                mvars = m.getVars()
                ubpen = [1.0] * m.numVars
                m.feasRelax(0, False, mvars, None, ubpen, None, None)
                print('solving...')
                m.optimize()
            # Extract Areas to Regions

        t4 = timer()

        print('Solving', t4 - t3, ' seconds')

        region = {'X': []}
        values = []
        test = m.getVars()
        for v in m.getVars():
            if v.x > 0:
                values.append((v.varName, v.x))
                if 'X' in v.varName:
                    a = v.varName[1:]
                    region['X'].append(int(a.split(',')[0][1:-1]))
        gdf = gdf.drop(['areas'], axis=1, errors='ignore')
        gdf['areas'] = np.where(gdf['id'].isin(region['X']), 1, 0)
        fig = gdf.plot(figsize=(10, 10), column='areas', edgecolor="black", facecolor="grey", cmap='OrRd')
        fig.set_title(title)
        t5 = timer()
        print('Displaying solution', t5 - t4, ' seconds')
        return region, ob_1.getValue(), t4 - t3

forsys_selected_areas, forsys_objective_values, forsys_time = shirabi(gdf_forsys,T,var_opt = [var_obj1], var_area = var_area, id_var = id_var,title="Optimal LUMs with Forsys departure indicator")
plt.show()
landfire_selected_areas, landfire_objective_values, landfire_time = shirabi(gdf_landfire,T,var_opt = [var_obj2], var_area = var_area, id_var = id_var,title="Optimal LUMs with Landfire departure indicator")
plt.show()
landclim_selected_areas, landclim_objective_values, landclim_time = shirabi(gdf_landclim,T,var_opt = [var_obj3], var_area = var_area, id_var = id_var,title="Optimal LUMs with Landclim departure indicator")
plt.show()

print("comparison")

print("forsys: "+ str(forsys_objective_values))
landfire_areas_on_forsys = [gdf_forsys[column_forsys].iloc[[i for i in landfire_selected_areas['X']]]*gdf_forsys['Acres'].iloc[[i for i in landfire_selected_areas['X']]]][0].sum()
print("landfire_areas_on_forsys: "+ str(landfire_areas_on_forsys))
landclim_areas_on_forsys = [gdf_forsys[column_forsys].iloc[[i for i in landclim_selected_areas['X']]]*gdf_forsys['Acres'].iloc[[i for i in landclim_selected_areas['X']]]][0].sum()
print("landclim_areas_on_forsys: "+ str(landclim_areas_on_forsys))

print("landfire: "+ str(landfire_objective_values))
landclim_areas_on_landfire = [gdf_landfire[column_landfire].iloc[[i for i in landclim_selected_areas['X']]]*gdf_landfire['Acres'].iloc[[i for i in landclim_selected_areas['X']]]][0].sum()
print("landclim_areas_on_landfire: "+ str(landclim_areas_on_landfire))
forsys_areas_on_landfire = [gdf_landfire[column_landfire].iloc[[i for i in forsys_selected_areas['X']]]*gdf_landfire['Acres'].iloc[[i for i in forsys_selected_areas['X']]]][0].sum()
print("landclim_areas_on_landfire: "+ str(forsys_areas_on_landfire))

print("landclim: "+ str(landclim_objective_values))
landfire_areas_on_landclim = [gdf_landclim[column_landclim].iloc[[i for i in landfire_selected_areas['X']]]*gdf_landclim['Acres'].iloc[[i for i in landfire_selected_areas['X']]]][0].sum()
print("landfire_areas_on_landclim: "+ str(landfire_areas_on_landclim))
forsys_areas_on_landclim = [gdf_landclim[column_landclim].iloc[[i for i in forsys_selected_areas['X']]]*gdf_landclim['Acres'].iloc[[i for i in forsys_selected_areas['X']]]][0].sum()
print("forsys_areas_on_landclim: "+ str(forsys_areas_on_landclim))

model_matrix = np.zeros((3,3))
model_matrix[0,0] = forsys_objective_values
model_matrix[0,1] = landfire_areas_on_forsys
model_matrix[0,2] = landclim_areas_on_forsys
model_matrix[1,0] = forsys_areas_on_landfire
model_matrix[1,1] = landfire_objective_values
model_matrix[1,2] = landclim_areas_on_landfire
model_matrix[2,0] = forsys_areas_on_landclim
model_matrix[2,1] = landfire_areas_on_landclim
model_matrix[2,2] = landclim_objective_values

print("Model matrix comparison")
print(model_matrix)
