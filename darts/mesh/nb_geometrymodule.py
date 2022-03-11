# For the class defintions here, only numpy is used:
import numpy as np
import numba as nb
from numba import float64, int32, types
from numba.typed import Dict
from numba.experimental import jitclass
# ------------------------------------------------------------
# Start matrix geometrical elements here: 3D objects
# Currently supported matrix
# ------------------------------------------------------------
""""
    Some definitions:
        - Nodes:    Vertices or points
        - Cells:    Control volumes
        - Face:     Sides of the control volume
        
    Most of the calculations regarding subdividing control volumes into tetrahedrons is taken from this paper:
    https://www.researchgate.net/publication/221561839_How_to_Subdivide_Pyramids_Prisms_and_Hexahedra_into_Tetrahedra
"""

kv_ty = (int32, int32[:])
spec = [
    ('volume', float64),
    ('depth', float64),
    ('centroid', float64[:]),
    ('nodes_to_cell', int32[:]),
    ('coord_nodes_to_cell', float64[:,:]),
    ('geometry_type', types.unicode_type),
    ('nodes_to_faces', types.DictType(*kv_ty)),
    ('permeability', float64[:]),
    ('prop_id', int32),
]

# The following parent class contains all the definitions for the (currently) supported geometric objects for
# unstructured reservoir (typically when imported from GMSH, but should be generalizable to any type of mesh):
@jitclass(spec)
class nbHexahedron:
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the parents class ControlVolume
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        """
        # Initialize some object variables:
        self.volume = 0.0
        self.depth = 0.0
        self.centroid = np.zeros(3)
        self.nodes_to_cell = nodes_to_cell
        self.coord_nodes_to_cell = coord_nodes_to_cell
        self.geometry_type = geometry_type
        self.nodes_to_faces = Dict.empty(*kv_ty)

        # This might be new for people not familiar to OOP. It is possible to call class methods which are defined below
        # as (and can be overloaded by child class) during the construction of a class:
        self.calculate_centroid()  # Class method which calculates the centroid of the control volume
        self.calculate_depth()  # Class method which calculates the depth (center) of the control volume
        self.calculate_nodes_to_face()  # Class method which finds the array containing the nodes of each face of the CV
        self.calculate_volume()  # Class method which calculates the volume of the CV

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_centroid(self):
        """
        Class method that calculates the centroid of the control volume (just the arithmic mean of the nodes coordinates
        :return:
        """
        self.centroid = np.sum(self.coord_nodes_to_cell, axis=0) / self.coord_nodes_to_cell.shape[0]
        return 0

    def calculate_depth(self):
        """
        Class method which calculates the depth of the particular control volume (at the  center of the volume)
        :return:
        """
        self.depth = np.abs(self.centroid[2])  # The class method assumes here that the third coordinate is the depth!
        return 0

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Top and bottom faces (rectangles)
        self.nodes_to_faces[0] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[4], self.nodes_to_cell[5]], dtype=np.int32)  # Bottom hexahedron
        self.nodes_to_faces[1] = np.array([self.nodes_to_cell[2], self.nodes_to_cell[3],
                                  self.nodes_to_cell[6], self.nodes_to_cell[7]], dtype=np.int32)  # Top hexahedron

        # Side faces (rectangles)
        self.nodes_to_faces[2] = np.array([self.nodes_to_cell[4], self.nodes_to_cell[5],
                                  self.nodes_to_cell[6], self.nodes_to_cell[7]], dtype=np.int32)  # Front hexahedron
        self.nodes_to_faces[3] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2], self.nodes_to_cell[3]], dtype=np.int32)  # Back hexahedron
        self.nodes_to_faces[4] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[3],
                                  self.nodes_to_cell[4], self.nodes_to_cell[7]], dtype=np.int32)  # Side hexahedron
        self.nodes_to_faces[5] = np.array([self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[5], self.nodes_to_cell[6]], dtype=np.int32)  # Side hexahedron
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split the hexahedron into five-tetrahedrons (see paper mentioned above for definitions and method):
        # Determine array with five-possible tetrahedrons (entries of array are nodes that belong to the CV):
        nodes_array_tetras = np.array([[4, 5, 1, 6],
                                       [4, 1, 0, 3],
                                       [4, 6, 3, 7],
                                       [1, 6, 3, 2],
                                       [4, 1, 6, 3]])

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3), dtype=np.float64)

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            # Calculate volume for current tetra and add to total volume:
            # Even new(er) and faster way:
            vec_edge_1 = local_coord[0] - local_coord[3]
            vec_edge_2 = local_coord[1] - local_coord[3]
            vec_edge_3 = local_coord[2] - local_coord[3]
            volume_tetra = np.abs(np.dot(vec_edge_1, np.cross(vec_edge_2, vec_edge_3))) / 6
            self.volume = self.volume + volume_tetra

@jitclass(spec)
class nbWedge:
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the parents class ControlVolume
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        """
        # Initialize some object variables:
        self.volume = 0.0
        self.depth = 0.0
        self.centroid = np.zeros(3)
        self.nodes_to_cell = nodes_to_cell
        self.coord_nodes_to_cell = coord_nodes_to_cell
        self.geometry_type = geometry_type
        self.nodes_to_faces = Dict.empty(*kv_ty)

        # This might be new for people not familiar to OOP. It is possible to call class methods which are defined below
        # as (and can be overloaded by child class) during the construction of a class:
        self.calculate_centroid()  # Class method which calculates the centroid of the control volume
        self.calculate_depth()  # Class method which calculates the depth (center) of the control volume
        self.calculate_nodes_to_face()  # Class method which finds the array containing the nodes of each face of the CV
        self.calculate_volume()  # Class method which calculates the volume of the CV

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_centroid(self):
        """
        Class method that calculates the centroid of the control volume (just the arithmic mean of the nodes coordinates
        :return:
        """
        self.centroid = np.sum(self.coord_nodes_to_cell, axis=0) / self.coord_nodes_to_cell.shape[0]
        return 0

    def calculate_depth(self):
        """
        Class method which calculates the depth of the particular control volume (at the  center of the volume)
        :return:
        """
        self.depth = np.abs(self.centroid[2])  # The class method assumes here that the third coordinate is the depth!
        return 0

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Top and bottom faces (triangles)
        self.nodes_to_faces[0] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2]], dtype=np.int32)  # Bottom wedge
        self.nodes_to_faces[1] = np.array([self.nodes_to_cell[3], self.nodes_to_cell[4],
                                  self.nodes_to_cell[5]], dtype=np.int32)  # Top wedge

        # Side faces (rectangles)
        self.nodes_to_faces[2] = np.array([self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[4], self.nodes_to_cell[5]], dtype=np.int32)  # Front wedge
        self.nodes_to_faces[3] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[3], self.nodes_to_cell[4]], dtype=np.int32)  # Side wedge
        self.nodes_to_faces[4] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3], self.nodes_to_cell[5]], dtype=np.int32)  # Side wedge
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split into three-tetrahedrons  (see paper mentioned above for definitions and method):
        # Determine array with five-possible tetrahedrons:
        nodes_array_tetras = np.array([[0, 1, 2, 4],
                                       [0, 3, 4, 5],
                                       [0, 2, 4, 5]])
        self.volume = 0

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3), dtype=np.float64)

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            vec_edge_1 = local_coord[0] - local_coord[3]
            vec_edge_2 = local_coord[1] - local_coord[3]
            vec_edge_3 = local_coord[2] - local_coord[3]
            volume_tetra = np.abs(np.dot(vec_edge_1, np.cross(vec_edge_2, vec_edge_3))) / 6
            self.volume = self.volume + volume_tetra
        return 0

@jitclass(spec)
class nbPyramid:
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id=-1):
        """
        Class constructor for the parents class ControlVolume
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        """
        # Initialize some object variables:
        self.volume = 0.0
        self.depth = 0.0
        self.centroid = np.zeros(3)
        self.nodes_to_cell = nodes_to_cell
        self.coord_nodes_to_cell = coord_nodes_to_cell
        self.geometry_type = geometry_type
        self.nodes_to_faces = Dict.empty(*kv_ty)

        # This might be new for people not familiar to OOP. It is possible to call class methods which are defined below
        # as (and can be overloaded by child class) during the construction of a class:
        self.calculate_centroid()  # Class method which calculates the centroid of the control volume
        self.calculate_depth()  # Class method which calculates the depth (center) of the control volume
        self.calculate_nodes_to_face()  # Class method which finds the array containing the nodes of each face of the CV
        self.calculate_volume()  # Class method which calculates the volume of the CV

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_centroid(self):
        """
        Class method that calculates the centroid of the control volume (just the arithmic mean of the nodes coordinates
        :return:
        """
        self.centroid = np.sum(self.coord_nodes_to_cell, axis=0) / self.coord_nodes_to_cell.shape[0]
        return 0

    def calculate_depth(self):
        """
        Class method which calculates the depth of the particular control volume (at the  center of the volume)
        :return:
        """
        self.depth = np.abs(self.centroid[2])  # The class method assumes here that the third coordinate is the depth!
        return 0

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Bottom faces (Quadrangle)
        self.nodes_to_faces[0] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2], self.nodes_to_cell[3]], dtype=np.int32)  # Bottom Quadrangle

        # Side faces (Triangle)
        self.nodes_to_faces[1] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[4]], dtype=np.int32)  # Top wedge
        self.nodes_to_faces[2] = np.array([self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[4]], dtype=np.int32)  # Top wedge
        self.nodes_to_faces[3] = np.array([self.nodes_to_cell[2], self.nodes_to_cell[3],
                                  self.nodes_to_cell[4]], dtype=np.int32)  # Top wedge
        self.nodes_to_faces[4] = np.array([self.nodes_to_cell[3], self.nodes_to_cell[0],
                                  self.nodes_to_cell[4]], dtype=np.int32)  # Top wedge
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split into two-tetrahedrons (see paper mentioned above for definitions and method):
        self.volume = 0
        # I think he determines the length of the edges in order to find the best shaped tetrahedrons (he uses the
        # resulting meshing in simulations, where orthogonality is important). For volume calculations the ordering
        # should not matter if I recall correct. (can always revert back changes to previous version!)
        nodes_array_tetras = np.array([[1, 2, 3, 4],
                                       [1, 3, 0, 4]])

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3), dtype=np.float64)

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            # Calculate volume for current tetra and add to total volume:
            # Even new(er) and faster way:
            vec_edge_1 = local_coord[0] - local_coord[3]
            vec_edge_2 = local_coord[1] - local_coord[3]
            vec_edge_3 = local_coord[2] - local_coord[3]
            volume_tetra = np.abs(np.dot(vec_edge_1, np.cross(vec_edge_2, vec_edge_3))) / 6
            self.volume = self.volume + volume_tetra

        return 0

@jitclass(spec)
class nbTetrahedron:
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id=-1):
        """
        Class constructor for the parents class ControlVolume
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        """
        # Initialize some object variables:
        self.volume = 0.0
        self.depth = 0.0
        self.centroid = np.zeros(3)
        self.nodes_to_cell = nodes_to_cell
        self.coord_nodes_to_cell = coord_nodes_to_cell
        self.geometry_type = geometry_type
        self.nodes_to_faces = Dict.empty(*kv_ty)

        # This might be new for people not familiar to OOP. It is possible to call class methods which are defined below
        # as (and can be overloaded by child class) during the construction of a class:
        self.calculate_centroid()  # Class method which calculates the centroid of the control volume
        self.calculate_depth()  # Class method which calculates the depth (center) of the control volume
        self.calculate_nodes_to_face()  # Class method which finds the array containing the nodes of each face of the CV
        self.calculate_volume()  # Class method which calculates the volume of the CV

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_centroid(self):
        """
        Class method that calculates the centroid of the control volume (just the arithmic mean of the nodes coordinates
        :return:
        """
        self.centroid = np.sum(self.coord_nodes_to_cell, axis=0) / self.coord_nodes_to_cell.shape[0]
        return 0

    def calculate_depth(self):
        """
        Class method which calculates the depth of the particular control volume (at the  center of the volume)
        :return:
        """
        self.depth = np.abs(self.centroid[2])  # The class method assumes here that the third coordinate is the depth!
        return 0

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        self.nodes_to_faces[0] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2]], dtype=np.int32)  # Top triangle
        self.nodes_to_faces[1] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[3]], dtype=np.int32)  # Side triangle
        self.nodes_to_faces[2] = np.array([self.nodes_to_cell[0], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3]], dtype=np.int32)  # Side triangle
        self.nodes_to_faces[3] = np.array([self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3]], dtype=np.int32)  # Side triangle
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Calculate area of tetrahedron shaped control volume:
        vec_edge_1 = self.coord_nodes_to_cell[0] - self.coord_nodes_to_cell[3]
        vec_edge_2 = self.coord_nodes_to_cell[1] - self.coord_nodes_to_cell[3]
        vec_edge_3 = self.coord_nodes_to_cell[2] - self.coord_nodes_to_cell[3]
        self.volume = np.abs(np.dot(vec_edge_1, np.cross(vec_edge_2, vec_edge_3))) / 6
        return 0
