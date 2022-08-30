import subprocess
import os
import numpy as np
import xml.etree.ElementTree as ET
import bs4 as bs
import geopandas as geop
from timeit import default_timer as timer
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from help_functions import geod_to_graph1, sink_ord, draw, queen, rook1, contiguous_patch_with_area_limit
import os

#TODO
# 1. configure Landtype parameters (with high ignition prob etc.)
# 2. set up the initial landscape. might be complicated
# 3. change the objective function to be maximizing too

landclim_input_folder = r"C:/Users/morit/rprojects/landclim/LandClim_Stanislaus_57LMU/1_Input"
landclim_exe = r"C:/Users/morit/rprojects/landclim/LandClim_Stanislaus_57LMU/0_LandClim_Model/LandClim.exe"
landclim_config = r"C:/Users/morit/rprojects/landclim/LandClim_Stanislaus_57LMU/1_Input/model-configuration_Stanislaus_LMU57_optimization_setup.xml"
landclim_output_folder = r"C:/Users/morit/rprojects/landclim/LandClim_Stanislaus_57LMU/2_Output"
management_config = os.path.join(landclim_input_folder, "harvest-parameters.xml")


def simulate_forest_fires(management_config,handled_lmus, thinning_percentage):
    def manipulate_harvest_parameters(management_config,handled_lmus, thinning_percentage):
        mytree = ET.parse(management_config)
        myroot = mytree.getroot()
        soup = bs.BeautifulSoup(open(management_config),'xml')
        harvests = soup.find_all('Harvest')
        for elem in harvests:
            limits = elem.find_all('ThinningDetailed')
            for limit in limits:
                if int(elem.ManagementAreaId.text) in handled_lmus:
                    limit.string = str(thinning_percentage)
                else:
                    limit.string = str(0)

        harvests = soup.find_all('Harvest')
        for elem in harvests:
            limits = elem.find_all('SpeciesLimits')
            if len(limits) == 2:
                limits[1].decompose()

        # Recursive function (do not call this method)
        def _get_prettified(tag, curr_indent, indent):
            out = ''
            for x in tag.find_all(recursive=False):
                if len(x.find_all()) == 0:
                    content = x.string.strip(' \n')
                else:
                    content = '\n' + _get_prettified(x, curr_indent + ' ' * indent, indent) + curr_indent

                attrs = ' '.join([f'{k}="{v}"' for k, v in x.attrs.items()])
                out += curr_indent + (
                    '<%s %s>' % (x.name, attrs) if len(attrs) > 0 else '<%s>' % x.name) + content + '</%s>\n' % x.name

            return out

        # Call this method
        def get_prettified(tag, indent):
            return _get_prettified(tag, '', indent)

        pretty_xml = get_prettified(soup, indent=4)

        f = open(management_config, "w")
        f.write("<?xml version='1.0' encoding='UTF-8'?>" + "\n")
        f.write(pretty_xml)
        f.close()

    manipulate_harvest_parameters(management_config,handled_lmus, thinning_percentage)

    bashCommand = landclim_exe + " " + landclim_config + " " + r"C:\Users\morit\rprojects\landclim\LandClim_Stanislaus_57LMU\1_Input\TreeInit.csv"

    simulation_run = subprocess.run(bashCommand.split(),
        check=True, text=True)

    # 4. cell count of burned area
    burned_cells_total = np.genfromtxt(os.path.join(landclim_output_folder,"fire_summary.csv"), delimiter=';', skip_header=True)[:,2].sum()
    print(burned_cells_total)

    return burned_cells_total


