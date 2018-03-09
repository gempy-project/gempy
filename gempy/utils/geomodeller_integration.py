"""Class definition for GeoModeller XML-Files
This version includes drillholes
Specific methods are defined for the uncertainty analysis (in combination
with Uncertainty_Obj module)

(c) J. Florian Wellmann, 2009-2013
"""

# try:
#     import elementtree.ElementTree as ET
# except ImportError:
#     try:
#         import etree.ElementTree as ET
#     except ImportError:
#         try:
#             import xml.etree.ElementTree as ET
#         except ImportError:
#             import ElementTree as ET
# import Latex_output_5 as LO
from pylab import *
import copy
import pandas as pn
import gempy as gp
import numpy as np

# python module to wrap GeoModeller XML file and perform all kinds of data
# procedures, e.g.:
# - Stochastic data modeling
# - TWT to depth conversion
# - Documentation module
#
# To Do:
# - exception handling and passing
# - log file?
# - check validity of xml-code, test, if past-processing with sax neccessary?
# - implement auto-documentation
# - clear-up spaghetti code!!!!! Check dependencies and other modules
#    to get a consistent lay-out

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

        # self.stratigraphic_column = self.get_stratigraphic_column()
        # self.faults = self.get_faults()
        #
        # self.series_info = self._get_series_fmt_dict()
        # self.series_distribution = self.get_series_distribution()
        #
        # self.fault_matrix = self.get_fault_matrix()

    def get_psc(self):
        """Returns the ProjectStratigraphicColumn tree element used for several data extractions."""
        return self.root.find("{" + self.xmlns + "}GeologicalModel").find("{"+self.xmlns+"}ProjectStratigraphicColumn")

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
                if c.tag == "{"+self.xmlns+"}Data":  # append formation names to list of formations
                    data[sn]["formations"].append(c.get("Name"))

                if c.tag == "{"+self.xmlns+"}InfluencedByFault":  # add fault influences
                    data[sn]["InfluencedByFault"].append(c.get("Name"))

                if c.tag == "{"+self.xmlns+"}PotentialField":

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

    def get_dataframes(self):
        """
        Extracts dataframe information from the self.data dictionary and returns GemPy-compatible interfaces and
        orientations dataframes.

        Returns:
            (tuple) of GemPy dataframes (interfaces, orientations)
        """
        interf_formation = []
        interf_series = []

        orient_series = []

        for i, s in enumerate(self.data.keys()):  # loop over all series
            if i == 0:
                coords = self.data[s]["interfaces"]
                grads = self.data[s]["gradients"]
            else:
                coords = np.append(coords, self.data[s]["interfaces"])
                grads = np.append(grads, self.data[s]["gradients"])

            for j, fmt in enumerate(self.data[s]["formations"]):
                for n in range(int(self.data[s]["interfaces_counters"][j, 0])):
                    interf_formation.append(fmt)
                    interf_series.append(s)

            for k in range(len(grads)):
                orient_series.append(s)

        interfaces = pn.DataFrame(coords, columns=['X', 'Y', 'Z'])
        interfaces["formation"] = interf_formation
        interfaces["series"] = interf_series

        orientations = pn.DataFrame(grads, columns=['G_x', 'G_y', 'G_z', 'X', 'Y', 'Z'])
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
                                z["Zmin"],  z["Zmax"]]).astype(float))

    def get_interfaces_df(self):
        """
        Extracts the interface data points stored in the GeoModeller xml file and returns it as a GemPy interfaces
        dataframe.

        Returns:
            pandas.DataFrame: InputData.interfaces dataframe
        """
        if self.root.find("{" + self.xmlns + "}Structural3DData") is None:
            print("No 3D data stored in given xml file.")
            return None
        else:
            fmts = [c.attrib["Name"] for c in self.root.find("{" + self.xmlns + "}Structural3DData")[0]]  # use formations
            xyzf = []

            for i, fmt in enumerate(fmts):  # loop over all formations
                for p in self.root[5][0][i]:  # loop over every point
                    entry = p[0].text.split(",")  # split the string by its seperator into coord strings
                    entry.append(fmt)

                    for s in self.series_info.keys():
                        if fmt in self.series_info[s]["formations"]:
                            series = s
                        else:
                            series = fmt

                    entry.append(series)
                    xyzf.append(entry)

            interfaces = pn.DataFrame(np.array(xyzf), columns=['X', 'Y', 'Z', "formation", "series"])
            interfaces[["X", "Y", "Z"]] = interfaces[["X", "Y", "Z"]].astype(float)
            return interfaces

    def get_orientation_df(self):
        """
        Extracts the orientation data points sotred in the GeoModeller xml file and returns it as a GemPy
        orientations dataframe.

        Returns:
            pandas.DataFrame: InputData.orientations dataframe
        """
        if self.root.find("{" + self.xmlns + "}Structural3DData") is None:
            print("No 3D data stored in given xml file.")
            return None
        else:
            fol = []
            for i, s in enumerate(self.root.find("{" + self.xmlns + "}Structural3DData")[1]):
                for c in self.root.find("{" + self.xmlns + "}Structural3DData")[1][i]:
                    entry = c[0][0].text.split(",")
                    entry.append(c.get("Dip"))
                    entry.append(c.get("Azimuth"))
                    # correct polarity from bool str to int
                    pol = c.get("NormalPolarity")
                    if pol == "true":
                        entry.append(1)
                    else:
                        entry.append(-1)
                    entry.append(s.get("Name"))
                    for series in self.series_distribution.keys():
                        if s.get("Name") in self.series_distribution[series]:
                            entry.append(series)

                    fol.append(entry)

            orientations = pn.DataFrame(np.array(fol), columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series'])
            orientations[["X", "Y", "Z", "dip", "azimuth"]] = orientations[["X", "Y", "Z", "dip", "azimuth"]].astype(float)
            orientations["polarity"] = orientations["polarity"].astype(int)

            return orientations

    def _get_series_fmt_dict(self):
        sp = {}
        for i, s in enumerate(self.stratigraphic_column):  # loop over all series
            fmts = []  # init formation storage list
            influenced_by = []  # init influenced by list
            for c in self.root.find("{" + self.xmlns + "}GeologicalModel").find("{"+self.xmlns+"}ProjectStratigraphicColumn")[i]:
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


# TODO think where this function should go
def read_vox(geo_data, path):
    """
    read vox from geomodeller and transform it to gempy format
    Returns:
        numpy.array: block model
    """
    import pandas as pn
    geo_res = pn.read_csv(path)

    geo_res = geo_res.iloc[9:]

    # ip_addresses = geo_res['nx 50'].unique()  # geo_data.interfaces["formation"].unique()
    ip_dict = geo_data.get_formation_number()
  #  ip_dict = geo_data.interfaces['formation number'].unique()

    geo_res_num = geo_res.iloc[:, 0].replace(ip_dict)
    block_geomodeller = np.ravel(geo_res_num.as_matrix().reshape(
        geo_data.resolution[0], geo_data.resolution[1], geo_data.resolution[2], order='C').T)
    return block_geomodeller


class GeomodellerClass:
    """Wrapper for GeoModeller XML-datafiles to perform all kinds of data
    manipulation and analysis on low level basis, e.g.:
    - Uncertainty Simulation
    - TWT to depth conversion
    - Data analysis, e.g. Auto-documentation"""

    def __init__(self):
        """Wrapper for GeoModeller XML-datafiles to perform all kinds of data
    manipulation and analysis on low level basis, e.g.:
    - Uncertainty Simulation
    - TWT to depth conversion
    - Data analysis, e.g. Auto-documentation"""

    def load_geomodeller_file(self, xml_file):
        self.xml_file_name = xml_file
        try:
            tree = ET.parse(xml_file)
        except IOError:
            print(("Can not open xml File " + xml_file + ": " + string_err))
            print ("Please check file name and directory and try again")
            raise IOError
        # safe tree on local varibale
        self.tree = tree
        # get rootelement
        self.rootelement = tree.getroot()
        # set other class variables
        self.xmlns = "http://www.geomodeller.com/geo"
        self.gml = "http://www.opengis.net/gml"

    def load_deepcopy_tree(self, deepcopy_tree):
        """load tree information from deepcopied tree into object"""
        self.tree = deepcopy_tree
        self.rootelement = deepcopy_tree.getroot()
        # set other class variables
        self.xmlns = "http://www.geomodeller.com/geo"
        self.gml = "http://www.opengis.net/gml"

    def deepcopy_tree(self):
        """create a deep copy of original tree to restore later, e.g. for uncertainty evaluation"""

        deepcopy_tree = copy.deepcopy(self.tree)
        deepcopy_tree.parent = None
        return deepcopy_tree

    def reload_geomodeller_file(self, deepcopy_tree):
        """restore original tree root from deep copy of orignial tree
        deep copy can be created (not automatically to save memory!) with
        self.deepcopy_tree()
        """
        try:
            self.tree = deepcopy_tree
            self.rootelement = self.tree.getroot()
        except NameError:
            print ("No deep copy of original tree available, please create with self.deepcopy_tree()")

    def get_model_extent(self):
        """get extent of model
        returns (x_min, x_max, y_min, y_max, z_min, z_max)
        and saves extent in self.x_min, self.x_max, etc.
        """
        extent_parent = self.rootelement.find("{"+self.xmlns+"}Extent3DOfProject")
        extentbox3D = extent_parent.find("{"+self.xmlns+"}ExtentBox3D")
        extent3D = extentbox3D.find("{"+self.xmlns+"}Extent3D")
        extent_xy = extent3D.find("{"+self.xmlns+"}ExtentXY")
        extent_z = extent3D.find("{"+self.xmlns+"}ExtentZ")
        self.x_min = float(extent_xy.get("Xmin"))
        self.x_max = float(extent_xy.get("Xmax"))
        self.y_min = float(extent_xy.get("Ymin"))
        self.y_max = float(extent_xy.get("Ymax"))
        self.z_min = float(extent_z.get("Zmin"))
        self.z_max = float(extent_z.get("Zmax"))
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    # def get_model_range(self):
    #     """get model range from model extent, e.g. for automatic mesh generation"""
    #     (x_min, x_max, y_min, y_max, z_min, z_max) = self.get_model_extent()
    #     from numpy import abs
    #     self.range_x = abs(x_max - x_min)
    #     self.range_y = abs(y_max - y_min)
    #     self.range_z = abs(z_max - z_min)
    #     return (self.range_x, self.range_y, self.range_z)

    def get_sections(self):
        """get sections out of rootelement, safe array with section elements
        in local variable"""
        sections_parent = self.rootelement.findall("{"+self.xmlns+"}Sections")[0]
        self.sections = sections_parent.findall("{"+self.xmlns+"}Section")
        return self.sections

    def get_faults(self):
        """get fault elements out of rootelement and safe as local list"""
        try:
            faults_parent = self.rootelement.findall("{"+self.xmlns+"}Faults")[0]
            self.faults = faults_parent.findall("{"+self.xmlns+"}Fault")
        except IndexError:
            print("No faults found in model")
        return self.faults

    def get_formations(self):
        """get formation elements out of rootelement and safe as local list"""
        formations_parent = self.rootelement.findall("{"+self.xmlns+"}Formations")[0]
        self.formations = formations_parent.findall("{"+self.xmlns+"}Formation")

    def get_stratigraphy_list(self, **kwds):
        """get project stratigraphy and return as list; lowermost formation: 1
        for GeoModeller dll access (this ist the formation number that is returned with
        the GetComputedLithologyXYZ function in the geomodeller dll
        optional keywords:
        out = string : set 'out' formation to this name (might be necessary for TOUGH2 simulation!)
        """
        series_list = []
        strati_column = self.rootelement.find("{"+self.xmlns+"}GeologicalModel").find("{"+self.xmlns+"}ProjectStratigraphicColumn")#.findall("{"+self.xmlns+"Series")
        series = strati_column.findall("{"+self.xmlns+"}Series")
        for s in series:
            data = s.findall("{"+self.xmlns+"}Data")
            for d in data:
                series_list.append(d.get("Name"))
        # append "out" as uppermost formation for "out values
        if "tough2" in kwds:
            if 'out' in kwds:
                series_list.append(kwds['out'])
            else:
                series_list.append("out")
        self.stratigraphy_list = series_list
        return series_list

    def get_section_names(self):
        """get all section names out of local variable self.sections"""
        # todo: return section names as dictionary with element and name?
        # test if self.sections defined, if not -> create
        try:
            self.sections
        except AttributeError:
            # print "Create sections Data array"
            self.get_sections()
        section_names = []
        for section in self.sections:
            section_names.append(section.get("Name"))
        return section_names

    def get_formation_names(self):
        """get formation names and return as list"""
        forms=[]
        try:
            self.formations
        except AttributeError:
            self.get_formations()
        for formation in self.formations:
            forms.append(formation.get("Name"))
        return forms

    def get_fault_names(self):
        """get fault names and return as list"""
        faults_list=[]
        try:
            self.faults
        except AttributeError:
            self.get_faults()
        for fault in self.faults:
            faults_list.append(fault.get("Name"))
        return faults_list

    def get_points_in_sections(self):
        """Create dictionary of all points (with obs-id) in all sections"""
        self.create_sections_dict()
        for sec in list(self.section_dict.keys()):
            forms = self.get_formation_point_data(self.section_dict[sec])
            if forms == None:
                print ("\t\t\tNo Formation Points in this section")
            else:
                for form in forms:
                    #print form.get("ObservationID")
                #    if form.get("ObservationID") == None: continue
                    data = form.find("{"+self.xmlns+"}Data")
                    print(("\nObsID = %s" % form.get("ObservationID")))
                    print(("\tFormation name\t= %s" % data.get("Name")))
                    element_point = form.find("{"+self.gml+"}LineString")
                    element_coords = element_point.find("{"+self.gml+"}coordinates")
                    tmp = element_coords.text.split(" ")
                    for tmp1 in tmp:
                        if tmp1 == '': continue
                        tmp_cds = tmp1.split(",")
                        print(("\tX = %.1f, Y = %.1f" % (float(tmp_cds[0]), float(tmp_cds[1]))))


                    fol = form.find("{"+self.xmlns+"}FoliationObservation")
                    if fol is not None:
                        print(("\tFoliation defined: azimuth = %.1f, dip = %.1f" % (float(fol.get("Azimuth")), float(fol.get("Dip")))))
                        # get position of foliation (yet another point)
                        pt = fol.find("{"+self.gml+"}Point")
                        c = pt.find("{"+self.gml+"}coordinates")
                        cds = c.text.split(",")
                        print(("\t\tX = %.1f, Y = %.1f" % (float(cds[0]), float(cds[1]))))

            print ("\n")
            print((80*"-"))
            print(("Foliations in section %s:" % sec))
            print((80*"-"))
            foliations = self.get_foliations(self.section_dict[sec])
            if foliations == None:
                print ("\t\t\tNo foliations in this section")
            else:
                for fol1 in foliations:
                    print(("\nObsID = %s" % fol1.get("ObservationID")))
                    data = fol1.find("{"+self.xmlns+"}Data")
                    fol = fol1.find("{"+self.xmlns+"}FoliationObservation")
                    print(( "\tFormation name\t= %s" % data.get("Name")))
                    print(("\tAzimuth = %.1f, dip = %.1f" % (float(fol.get("Azimuth")), float(fol.get("Dip")))))
                    pt = fol.find("{"+self.gml+"}Point")
                    c = pt.find("{"+self.gml+"}coordinates")
                    cds = c.text.split(",")
                    print(("\tX = %.1f, Y = %.1f" % (float(cds[0]), float(cds[1]))))
        return

    def get_formation_parameters(self):
        """read formation parameters; physical
        properties, density, th. cond etc... store in dict"""
        #
        # To do: re-write in a more elegant way and keep original
        # structure and key-words?
        #
        self.formation_params = {}
        try:
            self.formations
        except AttributeError:
            #            print "Create sections Data array"
            self.get_formations()
        for formation in self.formations:
            self.formation_params[formation.get("Name")] = {}
            geophys = formation.find("{"+self.xmlns+"}Geophysics")
            dens = geophys.find("{"+self.xmlns+"}DensityCompoundDistribution")
            dens_simple = dens.find("{"+self.xmlns+"}SimpleDistribution")
            self.formation_params[formation.get("Name")]["dens_mean"] = dens_simple.get("Mean")
            self.formation_params[formation.get("Name")]["dens_law"] = dens_simple.get("LawType")
            self.formation_params[formation.get("Name")]["dens_dev"] = dens_simple.get("Deviation")
            #             print geophys.getchildren()
            mag = geophys.find("{"+self.xmlns+"}RemanantMagnetizationCompoundDistribution")
            mag_simple = mag.find("{"+self.xmlns+"}SimpleDistributionVector")
            self.formation_params[formation.get("Name")]["mag"] = mag_simple.get("Mean")
            velocity = geophys.find("{"+self.xmlns+"}VelocityCompoundDistribution")
            velocity_simple = velocity.find("{"+self.xmlns+"}SimpleDistribution")
            self.formation_params[formation.get("Name")]["vel_mean"] = velocity_simple.get("Mean")
            self.formation_params[formation.get("Name")]["vel_law"] = velocity_simple.get("LawType")
            self.formation_params[formation.get("Name")]["vel_dev"] = velocity_simple.get("Deviation")
            # Thermal properties are only defined in newer versions of GeoModeller! thus check!

            th_cond = geophys.find("{"+self.xmlns+"}ThermalConductivityCompoundDistribution")
            if th_cond == None: continue
            th_cond_simple = th_cond.find("{"+self.xmlns+"}SimpleDistribution")
            self.formation_params[formation.get("Name")]["th_cond_mean"] = th_cond_simple.get("Mean")
            self.formation_params[formation.get("Name")]["th_cond_law"] = th_cond_simple.get("LawType")
            self.formation_params[formation.get("Name")]["th_cond_dev"] = th_cond_simple.get("Deviation")
            heat_prod = geophys.find("{"+self.xmlns+"}HeatProductionRateCompoundDistribution")
            heat_prod_simple = heat_prod.find("{"+self.xmlns+"}SimpleDistribution")
            self.formation_params[formation.get("Name")]["heat_prod_mean"] = heat_prod_simple.get("Mean")
            self.formation_params[formation.get("Name")]["heat_prod_law"] = heat_prod_simple.get("LawType")
            self.formation_params[formation.get("Name")]["heat_prod_dev"] = heat_prod_simple.get("Deviation")

            # same for other properties
        #    print th_cond
            #
            # !!! only simple distributions yet impl.
            #

    def create_fault_dict(self):
        """create dictionary for fault elements with names as keys"""
        # test if self.formations defined, if not -> create
        try:
            self.faults
        except AttributeError:
            print ("Create Formations list")
            self.get_faults()
        self.fault_dict = {}
        for fault in self.faults:
            self.fault_dict[fault.get("Name")] = fault
        return self.fault_dict

    def create_formation_dict(self):
        """create dictionary for formation elements with formation names as keys"""
        # test if self.formations defined, if not -> create
        try:
            self.formations
        except AttributeError:
            print ("Create formation dictionary")
            self.get_formations()
        self.formation_dict = {}
        for formation in self.formations:
            self.formation_dict[formation.get("Name")] = formation
        return self.formation_dict

    def create_sections_dict(self):
        """create dictionary for section elements with section names as keys
        (for easier use...)"""
        # test if self.sections defined, if not -> create
        try:
            self.sections
        except AttributeError:
            # print "Create sections dictionary"
            self.get_sections()
        self.section_dict = {}
        for section in self.sections:
            self.section_dict[section.get("Name")] = section
        return self.section_dict

    def get_foliations(self, section_element):
        """get all foliation data elements from a for section"""
        tmp_element = section_element.find("{"+self.xmlns+"}Structural2DData")
        # check in case there is no foliation defined in this section
        # tmp_element2 = tmp_element.find("{"+self.xmlns+"}Foliations")
        try:
            tmp_element2 = tmp_element.find("{"+self.xmlns+"}Foliations")
        except AttributeError:
            return None
        try:
            foliations = tmp_element2.findall("{"+self.xmlns+"}Foliation")
        except AttributeError:
            return None
        return foliations

    def get_foliation_dip(self, foliation_element):
        """get dip of foliation element"""
        return float(foliation_element.find("{"+self.xmlns+"}FoliationObservation").get("Dip"))

    def get_foliation_azimuth(self, foliation_element):
        """get dip of foliation element"""
        return float(foliation_element.find("{"+self.xmlns+"}FoliationObservation").get("Azimuth"))

    def get_folation_polarity(self, foliation_element):
        """get polarity of foliation element; return true if Normal Polarity"""
        return foliation_element.find("{"+self.xmlns+"}FoliationObservation").get("NormalPolarity")

    def get_foliation_coordinates(self, foliation_element):
        """get coordinates of foliation element"""
        element_fol = foliation_element.find("{"+self.xmlns+"}FoliationObservation")
        element_point = element_fol.find("{"+self.gml+"}Point")
        element_coords = element_point.find("{"+self.gml+"}coordinates")
        return str(element_coords.text)

    def get_formation_data(self, section_element):
        """not used any more! use get_formation_point_data(section_element) instead"""
        print ("not used any more! use get_formation_point_data(section_element) instead")
        return None

    def get_formation_point_data(self, section_element):
        """get all formation point data elements from a for section"""
        tmp_element = section_element.find("{"+self.xmlns+"}Structural2DData")
        # check in case there is no formation points defined in this section
        try:
            tmp_element2 = tmp_element.find("{"+self.xmlns+"}Interfaces")
        except AttributeError:
            return None
        return tmp_element2.findall("{"+self.xmlns+"}Interface")

    def get_name(self, section_element):
        """get the name of any section element (if defined)"""
        return section_element.find("{"+self.xmlns+"}Name")

    def get_interface_name(self, interface_element):
        """get name of interface, i.e. the formation"""
        return interface_element.find("{"+self.xmlns+"}Data").get("Name")

    def get_point_coordinates(self, point_elements, **args):
        """get the coordinates of a specific point memory locations"""
        point_list = list()

        for element in point_elements:
          name = element.find("{"+self.xmlns+"}Data").get("Name")
          #if args.has_key("if_name"):
          if "if_name" in args:
            if args["if_name"] != name: continue
          element_point = element.find("{"+self.gml+"}LineString")
          element_coords = element_point.find("{"+self.gml+"}coordinates")
          point_list.append((name+ " " + str(element_coords.text)))
        return point_list

    def change_formation_values_PyMC(self, **args):
        """ -So far is ready only to changes points in coordinates y. It is not difficult to add a new
        dimension

            - The dips and azimuth ObservationID must contain _d or _a respectively"""

        if "info" in args:
            section_dict = self.create_sections_dict()
            contact_points_dict = {}
            foliation_dict = {}
            for i in range(len(section_dict)):
                print(("\n\n\n", list(section_dict.keys())[i], "\n"))
                print ("Elements and their ID \n")
                contact_points = self.get_formation_point_data(list(section_dict.values())[i])

                try:
                    for contact_point in contact_points:
                        contact_points_dict[contact_point.get("ObservationID")] = contact_point
                        print((contact_point, contact_point.get("ObservationID")))
                except TypeError:
                    print ("No contact points in the section")
                #ObsID = contact_points.get("ObservationID")
                foliations = self.get_foliations(list(section_dict.values())[i])
                try:
                    for foliation in foliations:
                        # dictionary to access with azimth name
                        foliation_dict[foliation.get("ObservationID")+"_a"] = foliation
                        # dictionary to access with dip name
                        foliation_dict[foliation.get("ObservationID")+"_d"] = foliation
                        print((foliation, foliation.get("ObservationID")))

                except TypeError:
                    print ("No foliation in the section")
                try:
                    coord_interface = self.get_point_coordinates(contact_points)
                except TypeError:
                    print ("Element does not have iterable objects")

                print(("\nDictionaries:\n ", contact_points_dict, "\n", foliation_dict))

                print(("\n Contact points", contact_points, "\n", coord_interface, "\n"))

                print(("foliations" , foliations,  "\n"))
                try:
                    for i in range(len(foliations)):
                        print(("azimut:",self.get_foliation_azimuth(foliations[i])))
                        print(("dip",self.get_foliation_dip(foliations[i])))
                        print(("coordinates", self.get_foliation_coordinates(foliations[i])))
                except TypeError:
                    print ("No foliation in the section")
            return None
        #========================
        # change the stuff
        #=======================
        section_dict = self.create_sections_dict()
        contact_points_dict = {}
        foliation_dict = {}

        #Creation of dictionaries according to the ObservationID
        for i in range(len(section_dict)):
            # Contact points:
            try:
                contact_points = self.get_formation_point_data(list(section_dict.values())[i])
                for contact_point in contact_points:
                    contact_points_dict[contact_point.get("ObservationID")] = contact_point
            except TypeError:
                continue
            # Foliation Points
            try:
                foliations = self.get_foliations(list(section_dict.values())[i])
                for foliation in foliations:
                    # dictionary to access with azimth name
                    foliation_dict[foliation.get("ObservationID")+"_a"] = foliation
                    # dictionary to access with dip name
                    foliation_dict[foliation.get("ObservationID")+"_d"] = foliation
            except TypeError:
                continue

        # Passing our chain values:
            # Contact_points
        if "contact_points_mc" in args:
            for contac_point_mc in args["contact_points_mc"]:
                try:

                    element = contact_points_dict[str(contac_point_mc)]
                    element_point = element.find("{"+self.gml+"}LineString")
                    element_coords = element_point.find("{"+self.gml+"}coordinates")
                    point_list = element_coords.text.split(" ")
                    if point_list[-1] == '':
                        point_list = point_list[0:-1]

                    if len(point_list) == 1:
                        self.change_formation_point_pos(element, y_coord = contac_point_mc.value)
                    #Specific case of the Graben:
                    elif len(point_list) == 2:
                        self.change_formation_point_pos(element, y_coord = [contac_point_mc.value, contac_point_mc.value])
                    else:
                        print ("The lenght of the points to change does not fit with the number of changes in the input (>2)")
                except KeyError:
                    print(("The name of your PyMC variables (%s) does not agree with the ObservationID in the xml. Check misspellings." % str(contac_point_mc)))
                    continue
            # Azimuths
        if "azimuths_mc" in args:
            for azimuth_mc in args["azimuths_mc"]:
                #print azimuth_mc, type(azimuth_mc)
                try:
                    self.change_foliation(foliation_dict[str(azimuth_mc)], azimuth = str(azimuth_mc.value))
                except KeyError:
                    print(("The name of your PyMC variables (%s) does not agree with the ObservationID in the xml. Check misspellings." % str(azimuth_mc)))
                    continue
            # Dips
        if "dips_mc" in args:
            for dip_mc in args["dips_mc"]:
                try:
                    self.change_foliation(foliation_dict[str(dip_mc)], dip = str(dip_mc.value))
                except KeyError:
                    print(("The name of your PyMC variables (%s) does not agree with the ObservationID in the xml. Check misspellings." % str(dip_mc)))
                    continue


    # To do: vectorize this
    def change_formation_point_pos(self, element, **args):
        """change position of formation point in section element
        arguments:
        x_coord, y_coord : set to this coordinates
        add_x_coord, add_y_coord : add values to existing coordinates
        use if_name = and if_provenance = to add conditions!
        print_points = bool: print the list of points that will be modified (default: False)"""
        #    print "I am here"
        #print_points = kwds.get('print_points', False)
        prov = element.get("Provenance")

        name = element.find("{"+self.xmlns+"}Data").get("Name")

        #if args.has_key("if_name"):
        if "if_name" in args:
            if args["if_name"] != name: return
        # if args.has_key("if_provenance"):
        if "if_provenance" in args:
            if args["if_provenance"] != prov: return
        # element_fol = element.find("{"+self.xmlns+"}")
        element_point = element.find("{"+self.gml+"}LineString")
        element_coords = element_point.find("{"+self.gml+"}coordinates")
        point_list = element_coords.text.split(" ")
        #    print "poitn lits", point_list
        if point_list[-1] == '':
            point_list = point_list[0:-1]
        if len(point_list) > 1:
            x_coords = []
            y_coords = []
            if "print_points" in args:
                print (point_list)
            for point in point_list:
                # if point == '': continue
                a = point.split(',')
                #print a
                [x_coord, y_coord] = [float(a[0]), float(a[1])]
                x_coords.append(x_coord)
                y_coords.append(y_coord)
            # convert to arrays for calculation
            x_coords = array(x_coords)
            y_coords = array(y_coords)
            # Here  y_coord, and x_coord
            if "x_coord" in args:
                if shape(point_list) == shape(args["x_coord"]):
                #except TypeError:
                    x_coords = array(args["x_coord"])
                else:
                    print ("length of the points you want to change do not match with input dimensions")
            if "y_coord" in args:
                #print (args["y_coord"])
                #print array(args["y_coord"])
                if shape(point_list) == shape(args["y_coord"]):
                    y_coords = array(args["y_coord"])
                    #            print "ycoords", y_coords
                else:
                    print ("length of the points you want to change do not match with input dimensions")
            #print "Coordenates", x_coords, y_coords
            # Here add coords
            if "add_x_coord" in args:
                x_coords = x_coords + float(args["add_x_coord"])
            if "add_y_coord" in args:
                y_coords = y_coords + float(args["add_y_coord"])
            #    print y_coords
            # now, reconstruct output format strings
            out_text = ''
            for (i, x_coord) in enumerate(x_coords):
                out_text += "%f,%f " % (x_coords[i],y_coords[i])
            element_coords.text = out_text

        else:
            [x_coord, y_coord] = point_list[0].split(",")
            [x_coord, y_coord] = [float(x_coord), float(y_coord)]
            if "x_coord" in args:
                x_coord = float(args["x_coord"])
            if "y_coord" in args:
                y_coord = float(args["y_coord"])
            if "add_x_coord" in args:
                x_coord = x_coord + float(args["add_x_coord"])
            if "add_y_coord" in args:
                y_coord = y_coord + float(args["add_y_coord"])
            element_coords.text = "%f,%f" % (x_coord, y_coord)
        return None

    def change_foliation_polarity(self, element):
        """change polarity of foliation element"""
        if element.get("NormalPolarity") == "true":
            element.set("NormalPolarity", "false")
        else:
            element.set("NormalPolarity", "true")

    def change_foliation(self, element, **args):
        """change foliation data, argument one or more of: azimuth, dip,
        normalpolarity = true/false, x_coord, y_coord" or: add_dip, add_azimuth,
        add_x_coord, add_y_coord to add values to existing values!
        use if_name = and if_provenance = to add conditions!"""
        prov = element.get("Provenance")
        name = element.find("{"+self.xmlns+"}Data").get("Name")
        if "if_name" in args:
            if args["if_name"] != name: return
        if "if_provenance" in args:
            if args["if_provenance"] != prov: return
        element_fol = element.find("{"+self.xmlns+"}FoliationObservation")
        if "dip" in args:
            element_fol.set("Dip", args["dip"])
        if "azimuth" in args:
            element_fol.set("Azimuth", args["azimuth"])
        if "nomalpolarity" in args:
            element_fol.set("NormalPolarity", args["normalpolarity"])
        #
        # To Do: logfile, if dip >90, azi > 360, ...
        #
        if "add_dip" in args:
            dip_org = float(element_fol.get("Dip"))
            dip_new = dip_org + float(args["add_dip"])
            if dip_new > 90:
                dip_new = 180 - dip_new
                self.change_foliation_polarity(element_fol)
                azi_org = float(element_fol.get("Azimuth"))
                if azi_org < 180:
                    element_fol.set("Azimuth", str(azi_org+180))
                else:
                    element_fol.set("Azimuth", str(azi_org-180))
            element_fol.set("Dip", str(dip_new))
        if "add_azimuth" in args:
            azi_org = float(element_fol.get("Azimuth"))
            azi_new = azi_org + float(args["add_azimuth"])
            if azi_new > 360.0: azi_new -= 360
            element_fol.set("Azimuth", str(azi_new))
        element_point = element_fol.find("{"+self.gml+"}Point")
        element_coords = element_point.find("{"+self.gml+"}coordinates")
        [x_coord, y_coord] = element_coords.text.split(",")
        [x_coord, y_coord] = [float(x_coord), float(y_coord)]
        if "x_coord" in args:
            x_coord = float(args["x_coord"])
        if "y_coord" in args:
            y_coord = float(args["y_coord"])
        if "add_x_coord" in args:
            x_coord = x_coord + float(args["add_x_coord"])
        if "add_y_coord" in args:
            y_coord = y_coord + float(args["add_y_coord"])
        element_coords.text = "%f,%f" % (x_coord, y_coord)
        return None

    def twt_to_depth(self, sec_element, formula, **args):
        """Convert all data within a section from twt to depth (including
        orientation data!!
        Input: section element with data points, conversion function as
        string with 't' as placeholder for twt-time, e.g. '2 * (-t) ** 2 + 18 * (-t)'
        ATTENTION: check if t negative
        optional arguments:
        change_dip (boolean) : change dip angle in foliation data
        according to first derivative of twt-to-depth formula
        create_plot (boolean) : create summary plot with twt and converted depth
        for conversion formula control
        """
        # Idea: stoachstic apporach to twt -> depth conversion: apply several
        # possible formulae for quality estimation of resulting model?
        struct_data = sec_element.find("{"+self.xmlns+"}Structural2DData")
        interfaces = struct_data.find("{"+self.xmlns+"}Interfaces").findall("{"+self.xmlns+"}Interface")
        # save data in list to create a plot to check validity of conversion
        t_list = []
        v_list = []
        for interface in interfaces:
            gml_coords_element = interface.find("{"+self.gml+"}LineString").find("{"+self.gml+"}coordinates")
            # check for correct decimal, column (cs) and text separator (ts)
            ts = gml_coords_element.get("ts")
            cs = gml_coords_element.get("cs")
            data_array = gml_coords_element.text.split(ts)
            # check if last entry is empty (seems to happen often), if so: delete!
            # print gml_coords_element.text
            if data_array[-1] == "": del data_array[-1]
            text_new = ""
            # apply conversion formula for z-direction, i.e. dv;
            # no change in x-direction -> du = 0 (but: maybe neccessary for specific situations?)
            for entry in data_array:
                parts = entry.split(cs)
                # get original values
                # t as varibale, as defined in formula (input)
                t = float(parts[1])
                v_new = eval(formula)
                du = 0
                text_new += "%f%s%f%s" % (float(parts[0])+du, cs, v_new, ts)
                # append to list for check-plot
                t_list.append(-t)
                v_list.append(v_new)
            # print text_new
            gml_coords_element.text = text_new
            # print gml_coords_element.text
        # now change foliation position and dip angle! (check: given as argument?)
        # for dip angle: numerical determination of first derivative for
        # twt to depth conversion formula?
        if "change_dip" in args and args["change_dip"]:
            print ("change dip in seismic profile")

        # create check-plot
        if "create_plot" in args and args["create_plot"]:
            print ("Create plot with twt, converted depth pairs")
            plot(t_list,v_list,'.', label = formula)
            title("TWT to depth: Converted data points\nSection: " + sec_element.get("Name"))
            xlabel("TWT [ms]")
            ylabel("Converted depth [m]")
            legend()
            grid(True)
            show()

    def get_pole_points(self, element):
        # function to plot data points in geomodeller element
        u = []
        v = []
        poles = element.getiterator("{"+self.xmlns+"}Pole-Weight")
        for pole in poles:
            u.append(pole.get("U"))
            v.append(pole.get("V"))

        return (u,v)

    def plot_points(self, element):
        # plot u,v points in simple 2D plot
        (u,v) = self.get_pole_points(element)
        plot(u,v,'o')
        name = element.get("Name")
        title(name)
        savefig(name+"_points.png")

    def write_xml(self, save_dir, **args):
        """Write elementtree to xml-file
        arguments:
        print_path: Print the path where the xml is created"""
        # to do: filename (and directory?) as optional argument"""
        # flo, 10/2008
        #file = "changed_model.xml"
        tree_new = ET.ElementTree(self.rootelement)
        if "print_path" in args:
            print(("Write tree to file " + save_dir))
        tree_new.write(save_dir)
        #self.tree.write("changed_model.xml")
        self.tree.write(save_dir)

    def add_to_project_name(self, s):
        """add string s to project name, e.g. Number of uncertainty run"""
        name = self.rootelement.get("projectName")
        name_new = name + " " + s
        self.rootelement.set("projectName",name_new)

    def create_TOUGH_formation_names(self, **kwds):
        """create formation names that are compatible with format required by TOUGH,
        i.e. String of length 5
        returns and stores as dictionary with Geomodeller Names as key and TOUGH names as entry
        (self.tough_formation_names)
        simply cuts original formation name to a name length of 5;
        if cut name already exists: create new name, three first string values followed by two integers
        that are subsequently increased
        optional keywods:
        out = string : set out formation to this name (for TOUGH2: no leading spaces allowed! set to 5 chars!)
        """
        # import str
        self.tough_formation_names = {}
        # check if self.formation_names list already exists, if not: create
        try: self.formation_names
        except AttributeError:
            self.formation_names = self.get_stratigraphy_list(**kwds)
        # create list with tough names to check if name already exists
        tough_name_list = []
        for i,name in enumerate(self.formation_names):
            #    if self.formation_names[i] == '  out' or self.formation_names[i] == '  OUT':
            #   tough_name_list.append("OUT  ")
            #   continue
            cut_name = self.formation_names[i][0:5]
            if cut_name in tough_name_list:
                for j in range(100):
                    if "%3s%02d" % (cut_name[0:3],j) in tough_name_list:
                        continue
                    else:
                        cut_name = "%3s%02d" % (cut_name[0:3],j)
                        tough_name_list.append(cut_name)
                        break
            else:
                tough_name_list.append(cut_name)
            self.tough_formation_names[name] = "%5s" % str.upper(cut_name)
        return self.tough_formation_names


if __name__ == '__main__':
    print ("main")
