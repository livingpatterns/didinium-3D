"""
Analysis of the spacing between the two cilia bands of the Didium cell

Author: M. Ekin Ozberk
Date: August 8, 2023
"""


import os
import utils
import numpy as np
import open3d as o3d
from mayavi import mlab
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN

class DidiniumPC:

    def __init__(self, file_name) -> None:
       
        name, extension = os.path.splitext(file_name)

        self.file_name = name

        print(self.file_name)

        self.pc_all = None              # Didinium cell
        self.pc_channel_one = None      # Didinium filaments
        self.pc_channel_three = None    # Didinium cilia
        
        self.pc_channel_one_alligned = None 
        self.pc_channel_three_labelled = None

        self.first_cilia_band = None
        self.second_cilia_band = None

        self.first_cilia_band_alligned = None
        self.second_cilia_band_alligned = None

        self.sagittal_plane = None
        self.axial_plane = None

        self.pc_loaded = False

        if extension == ".txt":
            # Load the data from the file
            self.pc_all = np.transpose(np.loadtxt(file_name, delimiter=','))
            self.pc_channel_one = np.transpose(np.loadtxt(self.file_name + "_channel_one.txt", delimiter=','))
            self.pc_channel_three = np.transpose(np.loadtxt(self.file_name + "_channel_three.txt", delimiter=','))
            self.pc_loaded = True

            print("Point clouds are loaded.")

        elif extension == ".czi":
            img, shape = utils.load_czi_image(pth=file_name)
            self.pc_all = np.transpose(utils.generate_point_cloud(img=img, shape=shape, channel_mult=[1, 1, 3]))
            self.pc_channel_one = np.transpose(utils.generate_point_cloud(img=img, shape=shape, channel_mult=[1, 0, 0]))
            self.pc_channel_three = np.transpose(utils.generate_point_cloud(img=img, shape=shape, channel_mult=[0, 0, 5]))
            # 3 by n numpy arrays
            self.pc_loaded = True

            self.save_point_cloud()

            print("Image stacks are converted to point cloud and point clouds are saved to files.")

        else:
            print("File name is not valid.")


    def visualize_pc(self):
        utils.vis(data=self.pc_all, title="All PC", fig_id=1)
        utils.vis(data=self.pc_channel_one, title="Channel 1", fig_id=2)
        utils.vis(data=self.pc_channel_three, title="Channel 3", fig_id=3)
        mlab.show()


    def save_point_cloud(self):

        if 'raw_data' in self.file_name:
            parts = self.file_name.split('raw_data')
            out_file_path = ''.join(parts[:-1])
            out_file_path = os.path.join(out_file_path, 'output_data')

            if not os.path.exists(out_file_path):
                os.makedirs(out_file_path)

            out_file_path = out_file_path + parts[-1]

        else:
            out_file_path = self.file_name

        utils.write_numpy_array_to_txt(self.pc_all, out_file_path + ".txt")
        utils.write_numpy_array_to_txt(self.pc_channel_one, out_file_path + "_channel_one.txt")
        utils.write_numpy_array_to_txt(self.pc_channel_three, out_file_path + "_channel_three.txt")


    def _compute_sagittal_plane(self):
        a, b, c, d = utils.compute_projection_plane(self.pc_channel_one.T)
        self.sagittal_plane = np.array([a, b, c, d])


    def _label_cilia_bands(self):
        if self.sagittal_plane is None:
            self._compute_sagittal_plane()
            
        points = np.copy(self.pc_channel_three.T)
        
        projected_point_cloud = utils.project_point_cloud_onto_plane(points, self.sagittal_plane[0], self.sagittal_plane[1], self.sagittal_plane[2], self.sagittal_plane[3])
        
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
        self.first_cilia_band = points[second_largest_cluster_indices]
        self.second_cilia_band = points[largest_cluster_indices]


    def visualize_cilia_bands(self, visualize=True):
        if self.first_cilia_band is None or self.second_cilia_band is None:
            self._label_cilia_bands()
        
        # Set colors
        color1 = [0.0, 0.0, 0.5]   # Dark Blue 
        color2 = [1.0, 0.65, 0.0]  # Orange

        # Create Open3D point cloud objects
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()

        # Assign points and colors to the point cloud objects
        pcd1.points = o3d.utility.Vector3dVector(self.first_cilia_band)
        pcd1.paint_uniform_color(color1)

        pcd2.points = o3d.utility.Vector3dVector(self.second_cilia_band)
        pcd2.paint_uniform_color(color2)

        if visualize:
            o3d.visualization.draw_geometries([pcd1, pcd2], 'Cilia Bands')


    def compute_distance_between_cilia_bands(self):

        if self.first_cilia_band is None or self.second_cilia_band is None:
            self._label_cilia_bands()

        # Calculate the mean point (centroid) of each point cloud
        mean_point_cloud1 = np.mean(self.first_cilia_band, axis=0)
        mean_point_cloud2 = np.mean(self.second_cilia_band, axis=0)

        # Calculate the Euclidean distance between the mean points
        distance = np.linalg.norm(mean_point_cloud1 - mean_point_cloud2)

        print(f"Mean distance between two cilia bands: {distance}")


    def _compute_axial_plane(self):
        if self.first_cilia_band is None:
            self._label_cilia_bands()

        a, b, c, d = utils.compute_projection_plane(self.first_cilia_band)

        self.axial_plane = np.array([a, b, c, d])


    def align_axial_plane_to_z(self, visualize=True):
        if self.axial_plane is None:
            self._compute_axial_plane()

        if self.first_cilia_band is None or self.second_cilia_band is None:
            self._label_cilia_bands()

        k = self.axial_plane[:3]

        R = utils.rotation_matrix_from_vectors(-k, np.array([0, 0, 1]))

        center = np.mean(self.pc_channel_one.T, axis=0)
        self.pc_channel_one_alligned = o3d.geometry.PointCloud()
        self.pc_channel_one_alligned.points = o3d.utility.Vector3dVector(self.pc_channel_one.T)
        self.pc_channel_one_alligned = self.pc_channel_one_alligned.rotate(R, center=(center[0],center[1],center[2]))

        # Set colors
        color1 = [0.0, 0.0, 0.5]   # Dark Blue
        color2 = [1.0, 0.65, 0.0]  # Orange

        a, b, c, d = utils.compute_projection_plane(self.first_cilia_band)

        self.first_cilia_band_alligned = o3d.geometry.PointCloud()
        self.first_cilia_band_alligned.points = o3d.utility.Vector3dVector(utils.project_point_cloud_onto_plane(self.first_cilia_band, a, b, c, d))
        self.first_cilia_band_alligned = self.first_cilia_band_alligned.rotate(R, center=(center[0],center[1],center[2]))
        self.first_cilia_band_alligned.paint_uniform_color(color1)

        a, b, c, d = utils.compute_projection_plane(self.second_cilia_band)

        self.second_cilia_band_alligned = o3d.geometry.PointCloud()
        self.second_cilia_band_alligned.points = o3d.utility.Vector3dVector(utils.project_point_cloud_onto_plane(self.second_cilia_band, a, b, c, d))
        self.second_cilia_band_alligned = self.second_cilia_band_alligned.rotate(R, center=(center[0],center[1],center[2]))
        self.second_cilia_band_alligned.paint_uniform_color(color2)

        if visualize:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1e-5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([self.pc_channel_one_alligned, mesh_frame])


    def find_cilia_latitude(self, visualize=True):
        if self.pc_channel_one_alligned is None:
            self.align_axial_plane_to_z(visualize=False)

        sphere_pts =  np.asarray(self.pc_channel_one_alligned.points)
        fcb = np.asarray(self.first_cilia_band_alligned.points)
        scb = np.asarray(self.second_cilia_band_alligned.points)

        _, point_cloud_negative = utils.divide_point_cloud(sphere_pts, self.axial_plane)

        arr = np.vstack((sphere_pts, fcb, scb))
        sph = pyrsc.Sphere()
        # center, radius, _ = sph.fit(point_cloud_negative, thresh=1.0)

        radius, center = utils.sphereFit(point_cloud_negative[:,0], point_cloud_negative[:,1], point_cloud_negative[:,2])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.paint_uniform_color(np.array([173, 216, 230]) / 255.0)
        center = np.array(center)
        sphere.translate(center)

        if visualize:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1e-5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([self.pc_channel_one_alligned, self.first_cilia_band_alligned, self.second_cilia_band_alligned, sphere, mesh_frame])

        values = np.linspace(0, radius, 181)
        values = values[1:]

        center.shape = (3,)
        f = np.dot(fcb-center, np.array([0,0,-1]))
        max_index = np.searchsorted(values, np.mean(f))  
        min_index = np.searchsorted(values, np.min(f))  

        print(f"The first cilia array is between latitudes: {min_index} \u00b0 N and {max_index} \u00b0 N")

        s = np.dot(scb-center, np.array([0,0,1]))
        max_index = np.searchsorted(values, np.mean(s))  
        min_index = np.searchsorted(values, np.min(s))   
        print(f"The second cilia array is between latitudes: {min_index} \u00b0 N and {max_index} \u00b0 S")

