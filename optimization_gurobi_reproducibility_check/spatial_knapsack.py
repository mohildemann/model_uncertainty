import gurobipy as gp
from gurobipy import GRB, quicksum
import os
import geopandas as gpd
from help_functions import geod_to_graph1, contiguous_patch_with_area_limit, sink_ord

e = gp.Env(empty=True)

#setting up the Guroby environment with the license
e.setParam('USERNAME', r'jhildema@uni-muenster.de')
e.setParam('LICENSEID', 792691)
e.start()

#load the shapefile as geopandas dataframe
dir = r'C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_USBC\project_data\Stanislaus_LMU\Subset_LMU_57'
rfile = 'POD215_n57_LMU.shp'

gdf = gpd.read_file(os.path.join(dir, rfile))

# Compute columns defining the potential benefit.
# Here, the benefit is the variable res_depart.
# Res_depart quantifies the actual distance from the original state of the forest (fire resilient) to its current state.

gdf['benefit']=gdf['res_depart']*gdf['Acres']
gdf['id'] = gdf.index

# required column names of spatial dataframe
var_area = 'Acres'
id_var = 'id'
#area treshhold of 100 acres
T = 100

# define the contiguous neighborhood
G = geod_to_graph1(gdf,adj = 'queen')
# in M, the number of units belonging to the set (see M_Set) of contiguous spatial units are added that are within the patch treshhold of T.
# "This function returns a set of contiguos spatial units that comprise the largest patch allowed by the threshold T."
M_set = [contiguous_patch_with_area_limit(i,gdf,var_area,T) for i in gdf[id_var]]
M = [len(m) for m in M_set]
M1 = len(gdf)
R = [sink_ord(i,gdf) for i in range(len(gdf))]


X = sorted(list(G.nodes()))
Y = list(G.edges())
V = sorted(list(G.nodes()))
m = gp.Model("spatial_knapsack")

# Add variables
x = m.addVars(X, lb=0, vtype=GRB.BINARY, name='X_nodes')
y = m.addVars(Y, lb=0, vtype=GRB.CONTINUOUS, name='Y_edges')
v = m.addVars(V, lb=0, vtype=GRB.BINARY, name='V_nodes')

m.update()

# ! Rick Church made modifications to the model by adding 3 different constraints to Shirabe's model
# !  1) a neighborhood form of a clique like constraint that forces the sink of a given
# !     patch to have the lowest index of the LMUs selected for a patch
# !          INVOKED BY setting the parameter "neighborhood" to "true"
# !  2) a pairwise clique set enforcing conditions that a sink of a given patch mush have the
# !     lowest index of LMUs in a patch===> different form of condition (1)
# !	       INVOKED BY setting the parameter "pairwise" to "true"
# !  3) a constraint that is a form of a neighborhood constraint, that prevents LMUs that are
# !     too far from LMU i to be included in a patch if LMU i is a sink of the patch.
# !     this condition requires the solution of a number of "shortest/smallest area-based
# !     paths, one from each potential sink LMU. This increases preprocessing time when invoked.
# !          INVOKED BY setting the parameter "pathbased" to "true"
# !  There is also a logical limit on how high the parameter in the flow limit constraints needs to be.
# !  It is set as the number of smallest areas that equals or exceeds the area limit of tmax.
# !  4) There is a cnstraint that involves a property of what I term "adjacent-push." This
# !     constraint encourages at least one unit j that is adjacent to unit i to be one in value if
# !     LMU i has been fully selected at a value of one, as long as LMU i is not a sink. The
# !     impact of this constraint may yield a tighter LP bound as well as encourage a solution to be
# !     naturally integer instead of fractional.
# !          INVOKED by setting the parameter "adjacent_push" to "true"
# !==================================for areas larger than tmax:===========================
# !
# ! There can be LMU's that in themselves reach or exceed the threshold for a treatment project
# !  this version of the model allows such areas to be selected as a project or patch
# !  Note if a the value of tmax is 85% of the area of an LMU, then the contribution to the objective
# !  will be computed as 85% of the value of treating that LMU
# !
# !========================================================================================

# writeln ("   Path Analysis BEGINS " )
# !   ===========experimental section=====================
# !   for each area/lmu, solve a shortest path problem LMU i which minimizes
# !   the smallest area that can be added to a patch when also including LMU j
# !   If this area exceeds TMAX, then LMU i and LMU j can never be a part
# !   of a patch that has a sink at LMU i. To do this in a simple manner one needs to
# !   refer to an LMU not by its actual value, but by its internal index between 1..nn
# ! start at LMU(id(1)
# ! first find the largest and smallest LMU numbers

# !Conditions A1: New conditions of X variables and V variables,
# !if V(j) is a sink, then it must be the smallest index of the patch

# !conditions A2: This is a pairwise version of Condition A1
# !If a given LMU i has an index that is less than LMU j, then LMU i cannot be a
# ! part of the patch with a sink at LMU j
# ! at this time these constraints are written only for the adjacent units of i

# ! Path based regional constraints, such constraints
# ! limit the choice of LMUs when a given sink is selected
# ! there is one such constraint fro each LMU

forest_management = m.addVars(M1, vtype=GRB.BINARY, name="increase_fire_resilience")

#example_i = []
### Add constraints
# This constraint ensures that the inflow to a patch is greater or equal than the outflow
m.addConstrs((quicksum((y[i, j] - y[j, i]) for j in list(G[i])) >= x[i] - M[i] * v[i] for i in X),
             name='Shirabe_C1_positive_flow_constraint')
m.update()

# This constraint ensures there is only one sink per patch
m.addConstr(quicksum(v[i] for i in V) == 1, name='Shirabe_C2_one_sink_constraint')
m.update()

# "Constraints (3) ensure that there is no flow into any SU outside the region (where xi=0),
# and that the total inflow of any SU in the region (where xi=1) does not exceed M−1.
# This implies that there may be flow from an SU outside the region to an SU in the region.
# Even in such a case, although a sink may have to receive an extra amount of flow,
# the supply from each SU in the region still must reach the sink and the contiguity condition holds.
# this constraint defines that all patches can not be larger than threshold T"
m.addConstrs((quicksum(y[i, j] for j in list(G[i])) <= (M[i] - 1) * x[i] for i in X), name='Shirabe_C3_flow_excess')
m.update()

# This constraint ensures that the patches are below a certain threshold T. Default: 100 acres
m.addConstr(quicksum(x[i] * gdf.iloc[i][var_area] for i in X) <= T, name='C2_max_area')
m.update()

# This constraint ensures that the sink of a given
# patch has the lowest index of the LMUs selected for a patch
m.addConstrs((v[i] <= x[i] for i in X), name='Church_C1_sink_lowest_index')
m.update()

# unknown constraints

# Unknown constraint 1
# comment: Is this the constraint that ensures that X is 0 or 1?
m.addConstrs((x[i] <= 1 for i in X), name='probably_x_0_or_1')
m.update()

# Unknown constraint 2
# comment: is this Constraint 10 of the Shirabe paper?
m.addConstrs(((len(R[i]))*v[i]+quicksum(x[j] for j in R[i]) <= (len(R[i])) for i in X), name = 'C10')
m.update()

# Unknown constraint 3
m.addConstrs((v[i] + x[j] <= 1 for i in X for j in R[i]), name='C11')
m.update()

# Unknown constraint 4
m.addConstrs((x[i]-v[i] <= quicksum(x[j] for j in list(G[i])) for i in X), name = 'C14')
m.update()

ob_1 = quicksum(x[i] * gdf['benefit'][i] * gdf['Acres'][i] for i in X)
m.setObjective(ob_1, GRB.MAXIMIZE)
m.update()
m.optimize()