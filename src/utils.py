import os
import numpy as np
import open3d as o3d
from mayavi import mlab
from pathlib import Path
from aicspylibczi import CziFile
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from plyfile import PlyData, PlyElement


def load_czi_image(path):
    """
    Function for loading CZI image stacks 
    Input: data path
    Output: image, shape
    """
    czi = CziFile(path)

    # load image using CziFile lib
    img, shp = czi.read_image()
    shape = dict(shp)

    print(f"CZI file in {path} loaded!")

    return img, shape


def preprocess_image(slice, channel_mult, VALUE_THRESHOLD):
    """
    Each image has 3 channels and at each channel we have images of different protein structures of the cells.
    What we would like to do is to normalize each channel first then threshold in HSV space to determine where we 
    denote the strctural elements for didinium locations. 

    Inputs : 
        slice : current slice we are considering, an image
        channel_mult : an array with 3 values. 
        example use if channel_mult = [1, 1, 1] each channel is multiplied with 1.0 before thresholding operations
                    if channel_mult = [0, 0, 1] you are only taking the structure in the last channel 
                    if channel_mult = [0, 0, 5] maybe you are not getting enough data points to represent the channel
                    structure so you'd want to multiply the channel with a higher value so that more datapoints are above
                    the threshold for 3D representation of the channel. 
        VALUE_THRESHOLD : threshold for putting a point in location, should be between 0 and 1
                          if VALUE_THRESHOLD is high you'll have a more dense point cloud and vice versa
    """
    assert len(channel_mult) == 3, "channel_mult should be an array with 3 inputs."
    assert 0.0 < VALUE_THRESHOLD and VALUE_THRESHOLD < 1.0, "VALUE_THRESHOLD should be between 0 and 1"

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


def pixel_to_xyz(X, Y, Z, X_SCALE, Y_SCALE, Z_SCALE):
    """
    Calculate xyz location based on image scale and location on image
    X_SCALE, Y_SCALE, Z_SCALE : metadata
    """
    return np.array([X*X_SCALE, Y*Y_SCALE, Z*Z_SCALE])


def generate_point_cloud(img, shape, X_SCALE, Y_SCALE, Z_SCALE, 
                         channel_mult=[1, 1, 1], n = 3, VALUE_THRESHOLD=0.33) -> np.ndarray:
    """
    Generate point cloud from zstack 

    Inputs : 
        img : zstack
        shape : computed from zstack, see load_czi_image
        X_SCALE, Y_SCALE, Z_SCALE : metadata
        channel_mult : an array with 3 values. 
        example use if channel_mult = [1, 1, 1] each channel is multiplied with 1.0 before thresholding operations
                    if channel_mult = [0, 0, 1] you are only taking the structure in the last channel 
                    if channel_mult = [0, 0, 5] maybe you are not getting enough data points to represent the channel
                    structure so you'd want to multiply the channel with a higher value so that more datapoints are above
                    the threshold for 3D representation of the channel. 
        VALUE_THRESHOLD : threshold for putting a point in location, should be between 0 and 1
                          if VALUE_THRESHOLD is high you'll have a more dense point cloud and vice versa
    Returns:
        array of point cloud locations xyz as numpy array
    """
    point_cloud = []
    for z in range(shape["Z"]-n):
        curr_img = img[0,0,0,:,z,:,:]
        processed_img = preprocess_image(curr_img, channel_mult=channel_mult, VALUE_THRESHOLD=VALUE_THRESHOLD)
        for i in range(processed_img.shape[0]):
            for j in range(processed_img.shape[1]):
                if processed_img[i,j] != 0.0:
                    loc = pixel_to_xyz(X=j, Y=i, Z=z, X_SCALE=X_SCALE, Y_SCALE=Y_SCALE, Z_SCALE=Z_SCALE)
                    point_cloud.append(loc)

    print("Point Cloud is Generated!")

    return np.transpose(point_cloud) # transpose works the best


def visualize_pointcloud(data, title = None, fig_id = 1):
    # For visualizing point clouds
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


def write_numpy_array_to_ply(filename, numpy_array):
    """
    Assuming the numpy_array is an n by 3 array, create a PLY element 
    save numpy array to ply file
    """
    vertex = np.array([tuple(row) for row in numpy_array],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex')

    # Write the PLY file
    PlyData([el], text=True).write(filename)

    print("Point Cloud is saved to file!")


def write_numpy_array_to_txt(point_cloud, file_path):
    "Transpose the numpy array to a 3 by n array"
    numpy_array = point_cloud.T
    np.savetxt(file_path, numpy_array, delimiter=',')
    print("Point Cloud is saved to file!")


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


def label_cilia_bands(pointcloud, sagittal_plane):        
    "pointcloud = Channel 3"
    points = np.copy(pointcloud.T)
    
    projected_point_cloud = project_point_cloud_onto_plane(points, sagittal_plane[0], sagittal_plane[1], sagittal_plane[2], sagittal_plane[3])
    
    # Perform clustering using DBSCAN
    epsilon = 2e-6  # Adjust this value to control the density of points within a cluster
    min_points = 20  # Adjust this value to set the minimum number of points in a cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    cluster_labels = dbscan.fit_predict(projected_point_cloud)

    # Get the unique cluster labels and their respective counts
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    # Sort the labels based on the cluster sizes in descending order
    sorted_labels = unique_labels[np.argsort(label_counts)[::-1]]

    # Extract the labels of the two largest clusters
    largest_cluster_label = sorted_labels[0]
    second_largest_cluster_label = sorted_labels[1]

    # Filter the indices of the points belonging to the two largest clusters
    largest_cluster_indices = np.where(cluster_labels == largest_cluster_label)[0]
    second_largest_cluster_indices = np.where(cluster_labels == second_largest_cluster_label)[0]

    # Extract the points corresponding to the two largest clusters in the original point cloud
    first_cilia_band = points[second_largest_cluster_indices]
    second_cilia_band = points[largest_cluster_indices]

    return first_cilia_band, second_cilia_band


def visualize_cilia_bands(first_cilia_band, second_cilia_band):
    # define colors
    color1 = [0.0, 0.0, 0.5]   # Dark Blue 
    color2 = [1.0, 0.65, 0.0]  # Orange

    # Create Open3D point cloud objects
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    # Assign points and colors to the point cloud objects
    pcd1.points = o3d.utility.Vector3dVector(first_cilia_band)
    pcd1.paint_uniform_color(color1)

    pcd2.points = o3d.utility.Vector3dVector(second_cilia_band)
    pcd2.paint_uniform_color(color2)

    o3d.visualization.draw_geometries([pcd1, pcd2], 'Cilia Bands')

    
def xyz_to_spherical(x, y, z):
    """
    Convert XYZ coordinates to spherical coordinates.

    Args:
        x (numpy.ndarray): Array of X-coordinates.
        y (numpy.ndarray): Array of Y-coordinates.
        z (numpy.ndarray): Array of Z-coordinates.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Tuple of arrays containing spherical coordinates (r, theta, phi).
            r (numpy.ndarray): Array of distances from the origin (radius).
            theta (numpy.ndarray): Array of polar angles (inclination) in radians (0 <= theta <= pi).
            phi (numpy.ndarray): Array of azimuthal angles (azimuth) in radians (0 <= phi < 2*pi).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    phi[phi < 0] += 2 * np.pi

    return r, theta, phi


def xyz_to_cylindrical(x, y, z):
    """
    Convert XYZ coordinates to cylindrical coordinates.

    Args:
        x (numpy.ndarray): Array of X-coordinates.
        y (numpy.ndarray): Array of Y-coordinates.
        z (numpy.ndarray): Array of Z-coordinates.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Tuple of arrays containing cylindrical coordinates (r, theta, z).
            r (numpy.ndarray): Array of distances from the Z-axis (radius).
            theta (numpy.ndarray): Array of azimuthal angles (azimuth) in radians (0 <= theta < 2*pi).
            z (numpy.ndarray): Array of Z-coordinates.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    z = z

    theta[theta < 0] += 2 * np.pi

    return r, theta, z


def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------        
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix. 
                
    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.   
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.
        
    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        
        matrix = np.corrcoef(data_adjust.T)
        
    else:
        matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------        
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])
            
    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])
        
    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]
    
    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
        
    else:
        return point, normal    
    
def divide_point_cloud(point_cloud, a, b, c, d, tolerance=0.0):
    # Step 1: Define the Plane
    plane_normal = np.array([a, b, c])
    plane_D = d

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