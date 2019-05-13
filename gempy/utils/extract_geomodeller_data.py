
from pylab import *
import copy
import pandas as pn
import gempy as gp
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class ReadGeoModellerXML:
    def __init__(self, fp):
        """
        Reads in and parses a GeoModeller XML file to extract interface and orientation data and the overall model
        settings (e.g. extent and sequential pile). It uses ElementTree to parse the XML and the tree's root can
        be accessed using self.root for more direct access to the file.

        Todo: - extract faults

        Args:
            fp (str): Filepath for the GeoModeller xml file to be read.

        """
        self.tree = ET.ElementTree(file=fp)  # load xml as tree
        self.root = self.tree.getroot()

        self.xmlns = "http://www.geomodeller.com/geo"
        self.gml = "http://www.opengis.net/gml"

        self.extent = self._get_extent()
        self.data = self.extract_data()

        self.series = list(self.data.keys())
        self.stratigraphic_column, self.surface_points, self.orientations = self.get_dataframes()

        self.series_info = self._get_series_fmt_dict()

        self.faults = self.get_faults()

        self.series_distribution = self.get_series_distribution()

        self.fault_matrix = self.get_fault_matrix()

    def _get_extent(self):
        """
        Extracts model extent from ElementTree root and returns it as tuple of floats.

        Returns:
            tuple: Model extent as (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        xy = self.root[0][0][0][0].attrib
        z = self.root[0][0][0][1].attrib
        return tuple(np.array([xy["Xmin"], xy["Xmax"],
                               xy["Ymin"], xy["Ymax"],
                               z["Zmin"], z["Zmax"]]).astype(float))

    def extract_data(self):
        """
        Extracts relevant data from the GeoModeller XML file ElementTree root (self.root) and returns it as a dictionary.


        Returns:
            (dict): Data dictionary
        """
        data = {}
        for s in self.get_psc():
            sn = s.get("name")
            data[sn] = {}  # create a dict for each series
            data[sn]["formations"] = []
            data[sn]["InfluencedByFault"] = []
            data[sn]["relation"] = s.get("relation")  # add relation, whatever that is

            for c in s:
                if c.tag == "{" + self.xmlns + "}Data":  # append formation names to list of formations
                    data[sn]["formations"].append(c.get("Name"))

                if c.tag == "{" + self.xmlns + "}InfluencedByFault":  # add fault influences
                    data[sn]["InfluencedByFault"].append(c.get("Name"))

                if c.tag == "{" + self.xmlns + "}PotentialField":

                    data[sn]["gradients"] = []
                    data[sn]["interfaces"] = []
                    data[sn]["interfaces_counters"] = []
                    data[sn]["solutions"] = []
                    data[sn]["constraints"] = []

                    for cc in c:
                        # COVARIANCE
                        if cc.tag == "{" + self.xmlns + "}covariance":
                            data[sn]["covariance"] = cc.attrib

                        # GRADIENTS
                        if cc.tag == "{" + self.xmlns + "}Gradients":
                            for gr in cc:
                                data[sn]["gradients"].append([gr.get("Gx"), gr.get("Gy"), gr.get("Gz"),
                                                              gr.get("XGr"), gr.get("YGr"), gr.get("ZGr")])

                        # INTERFACES
                        if cc.tag == "{" + self.xmlns + "}Points":
                            for co in cc:
                                data[sn]["interfaces"].append([float(co[0].text), float(co[1].text), float(co[2].text)])

                        # INTERFACE COUNTERS
                        if cc.tag == "{" + self.xmlns + "}InterfacePoints":
                            for ip in cc:
                                data[sn]["interfaces_counters"].append([int(ip.get("npnt")), int(ip.get("pnt"))])

                        # CONSTRAINTS
                        if cc.tag == "{" + self.xmlns + "}Constraints":
                            for co in cc:
                                data[sn]["constraints"].append(float(co.get("value")))

                        # SOLUTIONS
                        if cc.tag == "{" + self.xmlns + "}Solutions":
                            for sol in cc:
                                data[sn]["solutions"].append(float(sol.get("sol")))

                        if cc.tag == "{" + self.xmlns + "}ModelFaults":
                            print('hey')

                    # convert from str to float
                    data[sn]["gradients"] = np.array(data[sn]["gradients"]).astype(float)
                    data[sn]["interfaces"] = np.array(data[sn]["interfaces"]).astype(float)
                    data[sn]["interfaces_counters"] = np.array(data[sn]["interfaces_counters"]).astype(float)
                    data[sn]["solutions"] = np.array(data[sn]["solutions"]).astype(float)

        return data

    def get_dataframes(self):

        strat_pile = dict.fromkeys(self.series)
        surface_points = pn.DataFrame()
        orientations = pn.DataFrame()

        for serie in self.series:
            strat_pile[serie] = self.data[serie]['formations']

            interf_s = self.data[serie].get('interfaces')
            orient_s = self.data[serie].get('gradients')

            formations = self.data[serie].get('formations')
            if interf_s is not None:
                interf = pn.DataFrame(columns=['X', 'Y', 'Z'], data=interf_s)
                interf['series'] = serie

                if len(formations) > 1:
                    interf_formations = []
                    for j, fmt in enumerate(formations):
                        for n in range(int(self.data[serie].get('interfaces_counters')[j, 0])):
                            interf_formations.append(fmt)
                    interf['formation'] = interf_formations

                else:
                    interf['formation'] = formations[0]

                surface_points = pn.DataFrame.append(surface_points, interf)

            if orient_s is not None:
                orient = pn.DataFrame(columns=['G_x', 'G_y', 'G_z', 'X', 'Y', 'Z'], data=orient_s)
                orient['series'] = serie
                orient['formation'] = formations[0]  # formation is wrong here but does not matter for orientations

                orientations = pn.DataFrame.append(orientations, orient)

        return strat_pile, surface_points, orientations

    def get_psc(self):
        """Returns the ProjectStratigraphicColumn tree element used for several data extractions."""
        return self.root.find("{" + self.xmlns + "}GeologicalModel").find(
            "{" + self.xmlns + "}ProjectStratigraphicColumn")


    def get_order_formations(self):
        order_formations = []
        for entry in self.series_distribution.values():
            if type(entry) is str:
                order_formations.append(entry)
            elif type(entry) is tuple:
                for e in entry:
                    order_formations.append(e)

        return order_formations

    def get_faults(self):
        """
        Extracts fault names from ElementTree root.

        Returns:
            tuple: Fault names (str) ordered as in the GeoModeller XML.
        """
        faults = []
        for c in self.root[2]:
            faults.append(c.get("Name"))
        return tuple(faults)

    def get_series_distribution(self):
        """
        Combines faults and stratigraphic series into an unordered dictionary as keys and maps the correct
        formations to them as a list value. Faults series get a list of their own string assigned as formation.

        Returns:
            (dict): maps Series (str) -> Formations (list of str)
        """
        series_distribution = {}
        for key in self.series_info.keys():
            fmts = self.series_info[key]["formations"]
            if len(fmts) == 1:
                series_distribution[key] = fmts[0]
            else:
                series_distribution[key] = tuple(fmts)

        for f in self.stratigraphic_column:
            if "Fault" in f or "fault" in f:
                series_distribution[f] = f

        return series_distribution

    def _get_series_fmt_dict(self):
        sp = {}
        for i, s in enumerate(self.stratigraphic_column):  # loop over all series
            fmts = []  # init formation storage list
            influenced_by = []  # init influenced by list
            for c in self.root.find("{" + self.xmlns + "}GeologicalModel").find(
                    "{" + self.xmlns + "}ProjectStratigraphicColumn")[i]:
                if "Data" in c.tag:
                    fmts.append(c.attrib["Name"])
                elif "InfluencedByFault" in c.tag:
                    influenced_by.append(c.attrib["Name"])
            # print(fmts)
            sp[s] = {}
            sp[s]["formations"] = fmts
            sp[s]["InfluencedByFault"] = influenced_by

        return sp


    def _where_do_faults_stop(self):
        fstop = {}
        for i, f in enumerate(self.root[2]):

            stops_on = []
            for c in self.root[2][i][2:]:
                stops_on.append(c.get("Name"))

            fstop[f.get("Name")] = stops_on

        return fstop

    def get_fault_matrix(self):
        nf = len(self.faults)
        fm = np.zeros((nf, nf))  # zero matrix of n_faults²
        fstop = self._where_do_faults_stop()
        for i, f in enumerate(self.faults):
            for fs in fstop[f]:
                j = np.where(np.array(self.faults) == fs)[0][0]
                fm[i, j] = 1

        return fm
