import os
import argparse

from didinium_pc import DidiniumPC

def main():

    parser = argparse.ArgumentParser(description="Didinium Cilia Bands Analysis")
    
    parser.add_argument('--file_name', required=True, help='Input video name (required)')
    parser.add_argument('--generate_pc', default=False, help='Compute point cloud (default: False)')

    args = parser.parse_args()

    file_name = args.file_name
    generate_pc = args.generate_pc

    if file_name.lower().endswith(".czi") or file_name.lower().endswith(".txt"):
        file_name = file_name[:-4]

    point_cloud_files_exist = False

    current_dir = os.getcwd()

    files = [os.path.join(current_dir, "2_3D_Cell_Analysis", "output_data",  fl) for fl in [file_name+".txt", file_name+"_channel_one.txt", file_name+"_channel_three.txt"] ]

    if all(os.path.exists(file) for file in files):
        point_cloud_files_exist = True

    if generate_pc or not point_cloud_files_exist:
        dpc = DidiniumPC(file_name=os.path.join(current_dir, "2_3D_Cell_Analysis", "raw_data",  file_name + '.czi'))
    else:
        dpc = DidiniumPC(file_name=os.path.join(current_dir, "2_3D_Cell_Analysis", "output_data",  file_name + '.txt'))
    
    dpc.visualize_pc()

    dpc.visualize_cilia_bands()

    dpc.compute_distance_between_cilia_bands()

    dpc.find_cilia_latitude()


if __name__ == "__main__":
    main()