"""
3D Cell analysis utils

This Python file provides all the functions used in the computation of point clouds and the analysis
of Didium cells captured as image stacks (CZI). 

To make changes to the project please alter the functions provided in this file and/or add to them.

Author: M. Ekin Ozberk
Date: August 8, 2023
"""


import os
import numpy as np
from mayavi import mlab
from math import sqrt
from pathlib import Path
from aicspylibczi import CziFile
import matplotlib.colors as colors
from plyfile import PlyData, PlyElement
from sklearn.decomposition import FastICA, PCA

Z_REF = -0.00028511699
X_SCALE = 1.3178822554981574e-7
Y_SCALE = 1.3178822554981574e-7
Z_SCALE = 9.9999999999999995e-7
VALUE_THRESHOLD = 0.33


def load_czi_image(pth):
    czi = CziFile(pth)

    czi.dims  # BSCZYX

    czi.size  # (1, 40, 4, 60, 1300, 1900)

    # Load the image slice I want from the file
    img, shp = czi.read_image()
    shape = dict(shp)

    print("CZI file is loaded!")

    return img, shape


def preprocess_image(slice, channel_mult=[1, 1, 1]):
    # Find the minimum and maximum values in the array
    min_value = slice.min()
    max_value = slice.max()
    
    # Normalize the array to the range [0, 1]
    slice = (slice - min_value) / (max_value - min_value)
    
    # Convert the 3D array to an image
    image = np.moveaxis(slice, 0, -1)

    # Multiply channel values to eliminate some channels if preffered
    image[:, :, 0] = image[:, :, 0] * channel_mult[0]
    image[:, :, 1] = image[:, :, 1] * channel_mult[1]
    image[:, :, 2] = image[:, :, 2] * channel_mult[2]

    # Convert RGB to HSV and extract the value (V) channel
    hsv_image = colors.rgb_to_hsv(image)
    value_channel = hsv_image[:, :, 2]
    value_channel_thresh = np.copy(value_channel)
    
    # This thresholding needs some fine tuning 
    value_channel_thresh[value_channel < VALUE_THRESHOLD] = 0.0
    value_channel_thresh[value_channel_thresh > 0.0] = 1.0
    
    return value_channel_thresh


def pixel_to_xyz(X,Y,Z):
    return np.array([X*X_SCALE, Y*Y_SCALE, Z*Z_SCALE])


def generate_point_cloud(img, shape, channel_mult=[1, 1, 1], n = 3):
    point_cloud = []
    for z in range(shape["Z"]-n):
        curr_img = img[0,0,0,:,z,:,:]
        processed_img = preprocess_image(curr_img, channel_mult=channel_mult)
        for i in range(processed_img.shape[0]):
            for j in range(processed_img.shape[1]):
                if processed_img[i,j] != 0.0:
                    loc = pixel_to_xyz(X=j, Y=i, Z=z)
                    point_cloud.append(loc)

    print("Point Cloud is Generated!")

    return np.array(point_cloud)


def write_numpy_array_to_ply(filename, numpy_array):
    # Assuming the numpy_array is an n by 3 array, create a PLY element
    vertex = np.array([tuple(row) for row in numpy_array],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex')

    # Write the PLY file
    PlyData([el], text=True).write(filename)

    print("Point Cloud is saved to file!")


def write_numpy_array_to_txt(point_cloud, file_path):
    # Transpose the numpy array to a 3 by n array
    numpy_array = point_cloud.T
    np.savetxt(file_path, numpy_array, delimiter=',')
    print("Point Cloud is saved to file!")


def vis(data, title = None, fig_id = 1):
    x = data[0,:] 
    y = data[1,:] 
    z = data[2,:] 

    if title is None:
        title = "Point Cloud"

    # Visualize it with mlab.surf
    col = z
    fig = mlab.figure(fig_id, bgcolor=(0, 0, 0), size=(128*5,128*5))
    mlab.points3d(x, 
                  y, 
                  z,
                  col,
                  mode="point",
                  colormap='spectral',
                  figure=fig,
                  )
    mlab.title(title, height=0.1, size=0.4)


# https://9to5answer.com/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def compute_projection_plane(data, equation=True):
    """
    Perform Independent Component Analysis on 3D data and return the plane description.

    Parameters:
        data (numpy.ndarray): 3D data with shape (N, 3) where N is the number of points.
        equation (bool): If True, returns the plane equation (a, b, c, d). If False, returns the point on the plane and the normal vector.

    Returns:
        tuple: The plane description in the specified format.
    """
    # Center the data to have zero mean
    data_centered = data - np.mean(data, axis=0)

    # Perform Independent Component Analysis
    ica = FastICA(n_components=2, tol=5e-3, max_iter=1000)
    ica.fit(data_centered)
    components = ica.components_

    # Choose the two ICA components to form the 2D plane
    plane_normal = np.cross(components[0], components[1])
    plane_normal /= np.linalg.norm(plane_normal)

    if equation:
        # Plane equation: ax + by + cz + d = 0
        a, b, c = plane_normal
        if a < 0:
            plane_normal = -plane_normal
        a, b, c = plane_normal
        d = -np.dot(plane_normal, np.mean(data, axis=0))
        return a, b, c, d
    else:
        return np.mean(data, axis=0), plane_normal


def project_point_cloud_onto_plane(point_cloud, a, b, c, d):
    """
    Project a point cloud onto a plane defined by ax + by + cz + d = 0.

    Parameters:
    - point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud.
    - a, b, c, d (float): Coefficients defining the plane equation ax + by + cz + d = 0.

    Returns:
    - numpy.ndarray: An Nx3 array representing the projected point cloud on the plane.
    """
    # Calculate the denominator for the distance formula (a^2 + b^2 + c^2).
    denominator = np.sqrt(a**2 + b**2 + c**2)

    # Calculate the distances from each point in the point cloud to the plane.
    distances = (a * point_cloud[:, 0] + b * point_cloud[:, 1] + c * point_cloud[:, 2] + d) / denominator

    # Calculate the projected points on the plane.
    x_proj = point_cloud[:, 0] - distances * a
    y_proj = point_cloud[:, 1] - distances * b
    z_proj = point_cloud[:, 2] - distances * c

    # Create the projected point cloud array.
    projected_point_cloud = np.column_stack((x_proj, y_proj, z_proj))

    return projected_point_cloud


def divide_point_cloud(point_cloud, plane_eq, tolerance=0.0):
    # Step 1: Define the Plane
    plane_normal = plane_eq[:3]
    plane_D = plane_eq[3]

    # Step 2: Extract Points on Each Side of the Plane
    points_on_positive_side = []
    points_on_negative_side = []

    for point in point_cloud:
        distance_to_plane = np.dot(plane_normal, point) + plane_D
        if abs(distance_to_plane) <= tolerance:
            # Consider points within the tolerance as on the plane
            points_on_positive_side.append(point)
            points_on_negative_side.append(point)
        elif distance_to_plane > 0:
            points_on_positive_side.append(point)
        else:
            points_on_negative_side.append(point)

    # Step 3: Store the Divided Point Clouds
    point_cloud_positive = np.array(points_on_positive_side)
    point_cloud_negative = np.array(points_on_negative_side)

    return point_cloud_positive, point_cloud_negative


# https://jekel.me/2015/Least-Squares-Sphere-Fit/
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = sqrt(t)

    return radius, np.array([C[0], C[1], C[2]])