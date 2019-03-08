
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

        Args:
            fp (str): Filepath for the GeoModeller xml file to be read.

        """
        self.tree = ET.ElementTree(file=fp)  # load xml as tree
        self.root = self.tree.getroot()

        self.xmlns = "http://www.geomodeller.com/geo"
        self.gml = "http://www.opengis.net/gml"

        self.extent = self._get_extent()
        self.data = self.extract_data()
        self.interfaces, self.orientations = self.get_dataframes()
        self.stratigraphic_column = self.get_stratigraphic_column()
        self.faults = self.get_faults()
        self.series_info = self._get_series_fmt_dict()
        self.series_distribution = self.get_series_distribution()

        # self.fault_matrix = self.get_fault_matrix()

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

                    # convert from str to float
                    data[sn]["gradients"] = np.array(data[sn]["gradients"]).astype(float)
                    data[sn]["interfaces"] = np.array(data[sn]["interfaces"]).astype(float)
                    data[sn]["interfaces_counters"] = np.array(data[sn]["interfaces_counters"]).astype(float)
                    data[sn]["solutions"] = np.array(data[sn]["solutions"]).astype(float)

        return data

    def get_psc(self):
        """Returns the ProjectStratigraphicColumn tree element used for several data extractions."""
        return self.root.find("{" + self.xmlns + "}GeologicalModel").find(
            "{" + self.xmlns + "}ProjectStratigraphicColumn")

    def get_dataframes(self):
        # print(self.data)
        """
        Extracts dataframe information from the self.data dictionary and returns GemPy-compatible interfaces and
        orientations dataframes.

        Returns:
            (tuple) of GemPy dataframes (interfaces, orientations)
        """
        interf_formation = []
        interf_series = []

        orient_series = []
        # print(self.data.keys())

        for i, s in enumerate(self.data.keys()):  # loop over all series
            # print(i,s)
            # print(self.data[s].get('interfaces'))
            # print(self.data[s])
            if i == 0:
                coords = self.data[s].get("interfaces")
                grads = self.data[s].get("gradients")

            else:
                coords = np.append(coords, self.data[s].get("interfaces"))
                grads = np.append(grads, self.data[s].get("gradients"))

            # print(coords)
            if self.data[s].get('interfaces') is not None:
                for j, fmt in enumerate(self.data[s]["formations"]):
                    # print(j,fmt)
                    # print(self.data[s].get("interfaces_counters"))
                    for n in range(int(self.data[s].get("interfaces_counters")[j, 0])):
                        # print(n)
                        interf_formation.append(fmt)
                        interf_series.append(s)

            gradssub = np.delete(grads, 0)
            lengrads = gradssub.reshape(-1, 6)
            # print(lengrads.shape)
        for k in range(lengrads.shape[0]):
            orient_series.append(s)

        coords = np.delete(coords, 0)  # because first value is none
        grads = np.delete(grads, 0)

        interfaces = pn.DataFrame(coords.reshape(-1, 3), columns=['X', 'Y', 'Z'])
        # print(len(interfaces), len(interf_formation))
        # print(len(interf_series))
        interfaces["formation"] = interf_formation
        interfaces["series"] = interf_series

        # print(grads)

        orientations = pn.DataFrame(grads.reshape(-1, 6), columns=['G_x', 'G_y', 'G_z', 'X', 'Y', 'Z'])
        # print(len(orientations),len(orient_series))
        orientations["series"] = orient_series

        dips = []
        azs = []
        pols = []
        for i, row in orientations.iterrows():
            dip, az, pol = gp.data_management.get_orientation((row["G_x"], row["G_y"], row["G_z"]))
            dips.append(dip)
            azs.append(az)
            pols.append(pol)

        orientations["dip"] = dips
        orientations["azimuth"] = azs
        orientations["polarity"] = pols

        return interfaces, orientations

    def get_stratigraphic_column(self):
        """
        Extracts series names from ElementTree root.

        Returns:
            tuple: Series names (str) in stratigraphic order.
        """
        stratigraphic_column = []
        for s in self.get_psc():
            stratigraphic_column.append(s.get("name"))
        return tuple(stratigraphic_column)

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
