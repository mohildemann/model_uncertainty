import geopandas as geop
from timeit import default_timer as timer
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from help_functions import geod_to_graph1, sink_ord, draw, queen, rook1, contiguous_patch_with_area_limit
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.express as px
from landclim_simulation import simulate_forest_fires

from pymoo.factory import get_problem, get_reference_directions
from pymoo.visualization.radviz import Radviz

e = gp.Env(empty=True)
#e.setParam('WLSACCESSID', '222f6f86-a313-4097-ab9e-1a3b5b8f5171')
#e.setParam('WLSSECRET', 'b76140af-e034-4231-91a4-8ac385ebed52')
e.setParam('USERNAME', r'jhildema@uni-muenster.de')
e.setParam('LICENSEID', 792691)
e.start()

dir = r'input\Subset_LMU_57'
rfile = 'LMU57_all_departures.shp'
column_forsys = "res_depart"
column_landfire = "dep_landfi"
column_landclim = "dep_landcl"

gdf_forsys = geop.read_file(os.path.join(dir, rfile))
gdf_forsys['benefit_forsys']=gdf_forsys[column_forsys]*gdf_forsys['Acres']
#normalize to min max
#gdf_forsys['benefit_forsys']=(gdf_forsys['benefit_forsys'] - gdf_forsys['benefit_forsys'].min()) / (gdf_forsys['benefit_forsys'].max() - gdf_forsys['benefit_forsys'].min())
gdf_forsys['id'] = gdf_forsys.index

gdf_landfire = geop.read_file(os.path.join(dir, rfile))
total_area = gdf_landfire['Acres'].sum()
gdf_landfire['benefit_landfire']=gdf_landfire[column_landfire]*gdf_landfire['Acres']
#gdf_landfire['benefit_landfire']=(gdf_landfire['benefit_landfire'] - gdf_landfire['benefit_landfire'].min()) / (gdf_landfire['benefit_landfire'].max() - gdf_landfire['benefit_landfire'].min())
gdf_landfire['id'] = gdf_landfire.index

gdf_landclim = geop.read_file(os.path.join(dir, rfile))
gdf_landclim['benefit_landclim']=gdf_landclim[column_landclim]*gdf_landclim['Acres']
#gdf_landclim['benefit_landclim']=(gdf_landclim['benefit_landclim'] - gdf_landclim['benefit_landclim'].min()) / (gdf_landclim['benefit_landclim'].max() - gdf_landclim['benefit_landclim'].min())
gdf_landclim['id'] = gdf_landclim.index

gdf_combined = gdf_forsys.copy()
gdf_combined['id'] = gdf_forsys['id']
gdf_combined['benefit_forsys'] = gdf_forsys['benefit_forsys']
gdf_combined['benefit_landfire'] = gdf_landfire['benefit_landfire']
gdf_combined['benefit_landclim'] = gdf_landclim['benefit_landclim']

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

def shirabi_weighted(gdf, T, var_opt, var_area, id_var, weights, max_values_for_normalization, min_max = "maximize", title = "No title given",plt_show = False):
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
    obj_1 = quicksum((x[i] * gdf[var_opt[0]][i] * weights[0]) for i in X)
    obj_2 = quicksum((x[i] * gdf[var_opt[1]][i] * weights[1])for i in X)
    obj_3 = quicksum((x[i] * gdf[var_opt[2]][i] * weights[2]) for i in X)

    obj_1 = obj_1 / (max_values_for_normalization[0])
    obj_2 = obj_2 / (max_values_for_normalization[1])
    obj_3 = obj_3 / (max_values_for_normalization[2])

    combined_objective_value = ((obj_1 + obj_2 + obj_3)/3)

    m.setObjective(combined_objective_value, min_max)
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

    t4 = timer()
    region = {'X': []}
    values = []

    for v in m.getVars():
        if v.x > 0:
            values.append((v.varName, v.x))
            if 'X' in v.varName:
                a = v.varName[1:]
                region['X'].append(int(a.split(',')[0][1:-1]))
    gdf = gdf.drop(['areas'], axis=1, errors='ignore')
    gdf['areas'] = np.where(gdf['id'].isin(region['X']), 1, 0)
    if plt_show is True:
        fig = gdf.plot(figsize=(10, 10), column='areas', edgecolor="black", facecolor="grey", cmap='OrRd')
        fig.set_title(title)
    t5 = timer()
    print('Displaying solution', t5 - t4, ' seconds')
    seperate_objective_values = [quicksum((x[i] * gdf[var_opt[0]][i]) for i in X).getValue()/max_values_for_normalization[0],quicksum((x[i] * gdf[var_opt[1]][i]) for i in X).getValue()/max_values_for_normalization[1], quicksum((x[i] * gdf[var_opt[2]][i]) for i in X).getValue()/max_values_for_normalization[2]]
    return region,seperate_objective_values, combined_objective_value.getValue(), t4 - t3

def shirabi(gdf, T, var_opt, var_area, id_var, min_max = "maximize", title = "No title given"):
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

solve = False
if solve is True:
    max_values_for_normalization = [312.89029183, 3947.94669924, 89.78111156]
    optimization_results_with_different_weights = {}
    l = 0
    for i in range(0,101):
        for j in range(0, 101):
            for k in range(0, 101):
                w1 = i / 100
                w2 = j / 100
                w3 = k / 100
                if w1 + w2 + w3 == 1.:
                    selected_areas, seperate_objective_values, combined_objective_value, time = shirabi_weighted(gdf_combined,
                              T,
                              var_opt = [var_obj1, var_obj2, var_obj3],
                              weights = [w1,w2,w3],
                              max_values_for_normalization = max_values_for_normalization,
                              var_area = var_area,
                              id_var = id_var,
                              title="Optimal LUMs with Forsys (weight = {w1}), Landfire (weight = {w2}) and Landclim (weight = {w3}) departure indicator".format(w1=w1,w2=w2,w3=w3))
                    optimization_results_with_different_weights[l] = {}
                    optimization_results_with_different_weights[l]['weights'] = [w1,w2,w3]
                    optimization_results_with_different_weights[l]['seperate_objective_values'] = seperate_objective_values
                    optimization_results_with_different_weights[l]['combined_objective_value'] = combined_objective_value
                    optimization_results_with_different_weights[l]['selected_areas'] = selected_areas
                    l += 1

    with open('output/optimization_results_with_different_weights.pkl', 'wb') as handle:
        pickle.dump(optimization_results_with_different_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('output/optimization_results_with_different_weights.pkl', 'rb') as handle:
    optimization_results_with_different_weights = pickle.load(handle)

point_cloud_obj1 = []
point_cloud_obj2 = []
point_cloud_obj3 = []
weights = []
point_cloud_combined = []
gdf_combined['frequencies'] = 0

map_configuration_repitition_count = []
for key, value in optimization_results_with_different_weights.items():
    point_cloud_obj1.append(value["seperate_objective_values"][0])
    point_cloud_obj2.append(value["seperate_objective_values"][1])
    point_cloud_obj3.append(value["seperate_objective_values"][2])
    point_cloud_combined.append(value["combined_objective_value"])
    weights.append(value["weights"])
    # count how often each unit is part of optimal solution
    for i in value["selected_areas"]['X']:
        gdf_combined.loc[i,"frequencies"] = gdf_combined.loc[i,"frequencies"] + 1
    # prepare count how often a whole map configuration is optimal
    map_configuration_repitition_count.append(value["selected_areas"]['X'])

#set into percentage
gdf_combined['frequencies'] = (gdf_combined['frequencies'] / len(point_cloud_combined)) * 100

fig = gdf_combined.plot(figsize=(10, 10), legend=True, column='frequencies', edgecolor="black", facecolor="grey", cmap='OrRd')
fig.set_title("Percentage of how often each LMU was part of optimal solution with different weights")
plt.show()

selected_areas, objective_value, time = shirabi(gdf_combined,
                          T,
                          var_opt = ["frequencies"],
                          var_area = var_area,
                          id_var = id_var,
                          title="Robust solution based on how often each LMU was part of optimal solution with different weights")
plt.show()

unique_solutions, count_unique_solutions = np.unique(np.array(map_configuration_repitition_count), return_counts=True)


plt.bar([i for i in range(len(unique_solutions.tolist()))], count_unique_solutions.tolist(), width= 0.2,tick_label=[i for i in range(len(unique_solutions.tolist()))])
plt.show()
#plot most frequent map configuration
gdf_combined['areas'] = np.where(gdf_combined['id'].isin(unique_solutions[1]), 1, 0)
fig = gdf_combined.plot(figsize=(10, 10), column='areas', edgecolor="black", facecolor="grey", cmap='OrRd')
fig.set_title("most frequqnet map representation")
plt.show()

unique_solutions = unique_solutions.tolist()

unique_solution_test = []
from landclim_simulation import simulate_forest_fires
point_cloud_obj1,point_cloud_obj2,point_cloud_obj3, weights = [], [], [], []
for key, value in optimization_results_with_different_weights.items():
    repr = [i for i in value["selected_areas"]['X']]
    if repr not in unique_solution_test:
        unique_solution_test.append(repr)
        point_cloud_obj1.append(value["seperate_objective_values"][0])
        point_cloud_obj2.append(value["seperate_objective_values"][1])
        point_cloud_obj3.append(value["seperate_objective_values"][2])
        weights.append(value["weights"])



burned_area = []
landclim_input_folder = r"input/landclim_simulation/1_Input"
landclim_exe = r"input/landclim_simulation/0_LandClim_Model/LandClim.exe"
landclim_config = r"input/landclim_simulation/1_Input/model-configuration_Stanislaus_LMU57_optimization_setup.xml"
landclim_output_folder = r"input/landclim_simulation/2_Output"
management_config = os.path.join(landclim_input_folder, "harvest-parameters.xml")
# for sol in unique_solutions:
#     burned_area.append(simulate_forest_fires(management_config, sol, 0.6))
#
#
#df = pd.DataFrame([weights, point_cloud_obj1,point_cloud_obj2,point_cloud_obj3, burned_area]).T


# with open(r'output/optimization_results_dataframe_with_simulation_results.pkl', 'wb') as handle:
#     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'output/optimization_results_dataframe_with_simulation_results.pkl', 'rb') as handle:
    df = pickle.load(handle)

df.columns =['weights', 'objective 1','objective 2','objective 3', 'burned_area']

fig = px.scatter_3d(df, x='objective 1', y='objective 2',z='objective 3', log_x=True, size_max=100)
fig.update_traces(textposition='top center')
fig.show()

pf = np.array([point_cloud_obj1,point_cloud_obj2,point_cloud_obj3]).T
plot = Radviz(title="Pseudo trade-off from different forest resilience departure meta-models",
              labels=["ForSys","Landfire", "LandClim"],
              endpoint_style={"s": 70, "color": "green"}, legend=True, label=df["burned_area"], annotate = df["objective 1"])
plot.set_axis_style(color="black", alpha=1.0)

plot.add(pf, c=df["burned_area"], s=20, cmap = "Reds", label = df["burned_area"])

plot.show()

max_values_for_normalization = [312.89029183, 3947.94669924, 89.78111156]
selected_areas, seperate_objective_values, combined_objective_value, time = shirabi_weighted(gdf_combined,
                                                                                             T,
                                                                                             var_opt=[var_obj1,
                                                                                                      var_obj2,
                                                                                                      var_obj3],
                                                                                             weights=[0.03, 0, 0.97],
                                                                                             max_values_for_normalization=max_values_for_normalization,
                                                                                             var_area=var_area,
                                                                                             id_var=id_var,
                                                                                             title="Optimal LUMs with central trade-off", plt_show=True)
plt.show()
