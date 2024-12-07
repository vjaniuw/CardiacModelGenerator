# CardiacModelGenerator


Overview

CardiacModelGenerator.py is a Python-based application designed for viewing slice overlays, converting pixels to universal coordinates, generating point clouds, and generating/enhancing tetrahedral meshes. Specifically, this is for cardiac models and uses MRI DICOM images and nifti masks.


How to Use

1. First install via 

```bash
pip install sphinx sphinx-markdown-builder

2.  Then run the following in python either in an IDE or via terminal: 

```bash 

from CardiacModelGenerator import CardiacModelGenerator 

CardiacModelGenerator.main()

__If you are using macOS or Apple Silicon__: 

__Use pythonw. If not wx will not work__

An example is below: 

![My Picture](iMarkdownPictures/Example1.png)

If successfully run, the following should appear: 
![My Picture](iMarkdownPictures/Example2.png)




Features

Image/Mask Viewer: Allows for a user to scroll through overlays of a mask and image Point Clouds: Can generate point cloud based on user inputs Universal Coordinates: Convers Mask/Image data to universal coordinates based on Dicom metadata Mesh: Allows for tetrahedral meshes from user inputs

Requirements

The script requires the following Python libraries:

wx numpy pydicom nibabel cv2 (OpenCV) random matplotlib pyvista Install dependencies using:

pip install wxpython numpy pydicom nibabel opencv-python matplotlib pyvista How to Use

Input Data: Prepare images in a folder (should be dicoms). Have masks as nifti.

Run the Script: Execute the script in your Python environment: CardiacModelGenerator.py

Interactive GUI: The script uses wx for GUI, allowing you to interactively select data and configure settings. Visualize Point Clouds: Choose from multiple colormaps and adjust parameters like point_size and tol. Functions

generate_point_cloud Generates a 3D point cloud from input coordinates and masks.

Parameters: coords1, coords2, coords3: Coordinate arrays. masks1, masks2, masks3: Corresponding mask arrays. whichmask: Mask value to extract (default: 1). tol: Tolerance for coordinate matching (default: 0.1). colormap_name: Colormap for visualization (default: "viridis"). point_size: Size of the points in the visualization (default: 5). Returns: A PyVista PolyData object representing the cleaned point cloud. Example Usage

Execute GUI. The user can:

Select Dicom Image Folder
User selects mask for that folder
User clicks view segmentation
User selects generate Point Cloud
User selects generate mesh
User selects fix mesh
User can look at quality by clicking mesh quality
Developed by vinayjani. Contributions and suggestions are welcome!

License

This project is licensed under the MIT License. See LICENSE for details.