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
raw_data_file = os.path.join(os.getcwd(), 'data', 'raw', 'Image4.czi')

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
                                        channel_mult=[1, 0, 0], 
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

# just the third channel
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
utils.visualize_pointcloud(data=pointcloud_1, title="Channel 1", fig_id=3)
utils.visualize_pointcloud(data=pointcloud_3, title="Channel 3", fig_id=4)
mlab.show()

"""
One advice I would have is would be after visualizing these point clouds 
I would save them in a file so I wouldn't have to recompute them everytime
because it takes a bit of time.
"""


