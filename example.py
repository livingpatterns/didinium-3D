import os
import sys
import PyQt5
import numpy as np
from mayavi import mlab

# First lets add ./src to path 
sys.path.append(os.path.join(os.getcwd(), 'src'))

import utils 

# Define Metadata for the data file 
# You should be able to get this info using ImageJ
Z_REF = -0.00028511699
X_SCALE = 1.3178822554981574e-7
Y_SCALE = 1.3178822554981574e-7
Z_SCALE = 9.9999999999999995e-7
VALUE_THRESHOLD = 0.33

# Lets define the raw data file (our czi stack)
raw_data_file = os.path.join(os.getcwd(), 'data', 'raw', 'Image9.czi')

# now load czi data using load_czi_image() funcion in utils
raw_data, data_shape = utils.load_czi_image(raw_data_file)


# now construct the whole structure
pointcloud = utils.generate_point_cloud(img=raw_data, 
                                        shape=data_shape, 
                                        X_SCALE=X_SCALE, 
                                        Y_SCALE=Y_SCALE, 
                                        Z_SCALE=Z_SCALE, 
                                        channel_mult=[1, 1, 3], 
                                        n = 3, 
                                        VALUE_THRESHOLD=VALUE_THRESHOLD
                                        )

# just the first channel 
pointcloud_1 = utils.generate_point_cloud(img=raw_data, 
                                        shape=data_shape, 
                                        X_SCALE=X_SCALE, 
                                        Y_SCALE=Y_SCALE, 
                                        Z_SCALE=Z_SCALE, 
                                        channel_mult=[2, 0, 0], 
                                        n = 3, 
                                        VALUE_THRESHOLD=VALUE_THRESHOLD
                                        )
# just the second channel 
pointcloud_2 = utils.generate_point_cloud(img=raw_data, 
                                        shape=data_shape, 
                                        X_SCALE=X_SCALE, 
                                        Y_SCALE=Y_SCALE, 
                                        Z_SCALE=Z_SCALE, 
                                        channel_mult=[0, 1, 0], 
                                        n = 3, 
                                        VALUE_THRESHOLD=VALUE_THRESHOLD
                                        )

# just the third channel : Ciliary bands
pointcloud_3 = utils.generate_point_cloud(img=raw_data, 
                                        shape=data_shape, 
                                        X_SCALE=X_SCALE, 
                                        Y_SCALE=Y_SCALE, 
                                        Z_SCALE=Z_SCALE, 
                                        channel_mult=[0, 0, 5], 
                                        n = 3, 
                                        VALUE_THRESHOLD=VALUE_THRESHOLD
                                        ) 

# Visualize point clouds
utils.visualize_pointcloud(data=pointcloud, fig_id=1)
utils.visualize_pointcloud(data=pointcloud_1, title="Channel 1", fig_id=2)
utils.visualize_pointcloud(data=pointcloud_2, title="Channel 2", fig_id=3)
utils.visualize_pointcloud(data=pointcloud_3, title="Channel 3", fig_id=4)
mlab.show()

"""
One advice I would have is would be after visualizing these point clouds 
I would save them in a file so I wouldn't have to recompute them everytime
because it takes a bit of time.
"""

point_cloud_path = os.path.join(os.getcwd(), 'data', 'pointclouds')
if not os.path.exists(point_cloud_path):
    os.mkdir(point_cloud_path)

utils.write_numpy_array_to_txt(pointcloud, os.path.join(point_cloud_path, 'pointcloud.txt'))
utils.write_numpy_array_to_txt(pointcloud_1, os.path.join(point_cloud_path, 'pointcloud_1.txt'))
utils.write_numpy_array_to_txt(pointcloud_2, os.path.join(point_cloud_path, 'pointcloud_2.txt'))
utils.write_numpy_array_to_txt(pointcloud_3, os.path.join(point_cloud_path, 'pointcloud_3.txt'))

"""
We want to approximate the plane of symmetry for the Didinium
We can use the channel_1 pointcloud and see in which plane it has the 
most variance to approximate the plane of symmetry. 
This might not be optimal for every data we have. 
"""
a, b, c, d = utils.compute_projection_plane(pointcloud_1)
sagittal_plane = np.array([a, b, c, d]) # Naming should be corrected




