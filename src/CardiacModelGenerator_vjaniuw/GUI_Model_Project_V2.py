#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:35:28 2024

@author: viv5149
"""

import wx
import glob
import numpy as np
import pydicom 
from pydicom import dcmread
import os 
import nibabel as nib
import cv2 
import random 
import matplotlib.cm as cm 
from matplotlib import pyplot as plt 
import pyvista as pv 

def generate_point_cloud(coords1=None, masks1=None, coords2=None, masks2=None, coords3=None, masks3=None, whichmask=1, tol=0.1, colormap_name="viridis", point_size=5):
    """
    Generate and visualize a point cloud from up to three pairs of coords and masks.

    Returns:
    - PyVista PolyData object for the cleaned point cloud.
    """
    # Initialize empty arrays to hold coordinates
    all_xcoords, all_ycoords, all_zcoords = [], [], []

    def process_coords_and_masks(coords, masks):
        seg_to_extract = np.zeros_like(masks)
        seg_to_extract[masks == whichmask] = 1
        xcoords = coords[:, :, 0, :][seg_to_extract == 1]
        ycoords = coords[:, :, 1, :][seg_to_extract == 1]
        zcoords = coords[:, :, 2, :][seg_to_extract == 1]
        return xcoords, ycoords, zcoords

    if coords1 is not None and masks1 is not None:
        x, y, z = process_coords_and_masks(coords1, masks1)
        all_xcoords.append(x), all_ycoords.append(y), all_zcoords.append(z)

    if coords2 is not None and masks2 is not None:
        x, y, z = process_coords_and_masks(coords2, masks2)
        all_xcoords.append(x), all_ycoords.append(y), all_zcoords.append(z)

    if coords3 is not None and masks3 is not None:
        x, y, z = process_coords_and_masks(coords3, masks3)
        all_xcoords.append(x), all_ycoords.append(y), all_zcoords.append(z)

    if not all_xcoords:
        raise ValueError("At least one pair of coords and masks must be provided.")

    xcoords = np.concatenate(all_xcoords)
    ycoords = np.concatenate(all_ycoords)
    zcoords = np.concatenate(all_zcoords)

    ptCloud = np.column_stack((xcoords, ycoords, zcoords))
    z_values = ptCloud[:, 2]
    norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    cmap = plt.get_cmap(colormap_name)
    colormap = cmap(norm)
    rgb_colors = (colormap[:, :3] * 255).astype(np.uint8)

    point_cloud = pv.PolyData(ptCloud)
    point_cloud["Colors"] = rgb_colors
    point_cloud_cleaned = point_cloud.clean(tolerance=tol)

    # Non-blocking PyVista plot with a title
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud_cleaned, scalars="Colors", rgb=True, point_size=point_size)
    plotter.add_title("Point Cloud Visualization", font_size=20)
    plotter.show(interactive_update=True)

    return point_cloud_cleaned


def generate_tetra_mesh(point_cloud_cleaned):
    """
    Generate a tetrahedral mesh from the cleaned point cloud and visualize it.

    Parameters:
    - point_cloud_cleaned: PyVista PolyData object representing the cleaned point cloud.

    Returns:
    - PyVista UnstructuredGrid object representing the tetrahedral mesh.
    """
    grid = point_cloud_cleaned.delaunay_3d()
    grid = grid.elevation()  # Add elevation scalar for visualization

    # Non-blocking PyVista plot with a title
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="Elevation", cmap="hot", show_edges=True, opacity=0.7)
    plotter.add_title("Tetrahedral Mesh Visualization", font_size=20)
    plotter.show(interactive_update=True)

    return grid

def clean_tetra_mesh(grid, subdivisions=2, poisson_iterations=10, clean_tolerance=0.001, quality_threshold=1e-5):
    """
    Clean and refine a tetrahedral mesh.

    Parameters:
    - grid: PyVista UnstructuredGrid object representing the tetrahedral mesh.
    - subdivisions: Number of subdivisions to refine the mesh.
    - poisson_iterations: Number of smoothing iterations.
    - clean_tolerance: Distance threshold for cleaning.
    - quality_threshold: Minimum area threshold for valid triangles.

    Returns:
    - PyVista PolyData object representing the cleaned surface.
    """
    # Extract the surface from the volumetric mesh
    surface = grid.extract_surface()

    # Subdivide the mesh
    subdivided_surface = surface.subdivide(nsub=subdivisions)

    # Smooth the subdivided surface
    smoothed_surface = subdivided_surface.smooth(n_iter=poisson_iterations)

    # Clean and repair the surface
    cleaned_surface = smoothed_surface.clean(tolerance=clean_tolerance)

    # Filter out degenerate triangles
    cell_quality = cleaned_surface.compute_cell_sizes()["Area"]
    valid_cells = cell_quality > quality_threshold
    filtered_surface = cleaned_surface.extract_cells(np.where(valid_cells)[0])

    return filtered_surface

def get_cell_quality(final_volumetric_mesh): 
    """Compute and visualize cell quality for a tetrahedral mesh."""
    qual = final_volumetric_mesh.compute_cell_quality(quality_measure='scaled_jacobian')
    plotter = pv.Plotter()  # Create a Plotter instance
    plotter.add_mesh(qual, scalars='CellQuality', show_edges=True)  # Add the mesh
    plotter.show(interactive_update=True)  # Display the interactive plot
    
    return qual



class CardiacMeshalyzer(wx.Frame): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        
        # Global variables for DICOM and segmentation data
        self.dicom_data = {}  # Dictionary to store DICOM data by view number
        self.segmentation_data = {}  # Dictionary to store segmentation data by view number
        
        self.SetTitle("Cardiac Meshalyzer")
        self.Maximize(True)  # Start in full-screen mode
        
        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.sizer)
        
        # Menu Bar 
        self.setup_menu_bar() 
        
        # Container for pages
        self.page_container = wx.Panel(self.panel)
        self.page_sizer = wx.BoxSizer(wx.VERTICAL)
        self.page_container.SetSizer(self.page_sizer)
        self.sizer.Add(self.page_container, 1, wx.EXPAND)
        
        # Add pages
        self.pages = {}
        self.current_page = None
        self.add_page("Start Page", StartPage)
        self.add_page("Home Page", HomePage)
        
        
        # Show the start page initially
        self.show_page("Start Page")
        
        self.Show() 

    def vinayDicomSeries(self, folder_path):
        """Process DICOM series in the given folder."""
        Images = glob.glob(folder_path + '/*.dcm')
        if not Images:
            return None
        
        Images = np.array(Images)
        Images = np.sort(Images)
        out_images = np.ndarray((len(Images),), dtype=object)
        for ii in range(len(Images)):
            out_images[ii] = dcmread(Images[ii])
    
        Num_Slices = len(out_images)
        Volume = np.zeros((out_images[0].Rows, out_images[0].Columns, Num_Slices))
        Coords = np.zeros((out_images[0].Rows, out_images[0].Columns, 3, Num_Slices))
    
        for ii in range(Num_Slices):
            Volume[:, :, ii] = np.array(out_images[ii].pixel_array)
            orientation_mat = np.array(out_images[ii].ImageOrientationPatient)
            Yxyz = orientation_mat[0:3]
            Xxyz = orientation_mat[3:]
            dx = np.array(out_images[ii].PixelSpacing[1])
            dy = np.array(out_images[ii].PixelSpacing[0])
            pos = np.array(out_images[ii].ImagePositionPatient)
    
            M = np.array([
                [Xxyz[0] * dy, Yxyz[0] * dx, 0, pos[0]],
                [Xxyz[1] * dy, Yxyz[1] * dx, 0, pos[1]],
                [Xxyz[2] * dy, Yxyz[2] * dx, 0, pos[2]],
                [0, 0, 0, 1]
            ])
    
            # Create a grid of (jj, kk) indices
            jj_indices, kk_indices = np.meshgrid(np.arange(out_images[ii].Columns), np.arange(out_images[ii].Rows), indexing='ij')
    
            # Reshape jj and kk to form the input array for transformation
            grid_points = np.stack([jj_indices.ravel() - 1, kk_indices.ravel() - 1, np.zeros(jj_indices.size), np.ones(jj_indices.size)], axis=1).T
    
            # Perform the matrix multiplication for all points in one step
            transformed_points = np.matmul(M, grid_points)
    
            # Reshape the result to fit Coords structure
            Px, Py, Pz = transformed_points[:3]
            Coords[:, :, 0, ii] = Px.reshape(out_images[ii].Columns, out_images[ii].Rows).T
            Coords[:, :, 1, ii] = Py.reshape(out_images[ii].Columns, out_images[ii].Rows).T
            Coords[:, :, 2, ii] = Pz.reshape(out_images[ii].Columns, out_images[ii].Rows).T
    
        return Volume, Coords, out_images


    def get_masks(self, mask_path):
        """Load segmentation masks."""
        

        masks = nib.load(mask_path)
        masks = masks.get_fdata() 
        
        return masks
        
    def setup_menu_bar(self): 
        menubar = wx.MenuBar()
        
        # Darker menu bar for macOS-like dark mode
        menubar.SetBackgroundColour(wx.Colour(30, 30, 30))  # Darker shade
        menubar.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
    
        # File Menu 
        file_menu = wx.Menu()
    
        # Load Image View menu items
        load_image_view_1 = file_menu.Append(wx.ID_ANY, "Load Image View 1")
        load_segmentation_view_1 = file_menu.Append(wx.ID_ANY, "Load Segmentation View 1")
        load_image_view_2 = file_menu.Append(wx.ID_ANY, "Load Image View 2")
        load_segmentation_view_2 = file_menu.Append(wx.ID_ANY, "Load Segmentation View 2")
        load_image_view_3 = file_menu.Append(wx.ID_ANY, "Load Image View 3")
        load_segmentation_view_3 = file_menu.Append(wx.ID_ANY, "Load Segmentation View 3")
    
        file_menu.AppendSeparator()
    
        # Save and clear options
        save_point_cloud = file_menu.Append(wx.ID_ANY, "Save Point Cloud")
        save_tetra_cloud = file_menu.Append(wx.ID_ANY, "Save Tetra Cloud")
        clear_all_files = file_menu.Append(wx.ID_ANY, "Clear All Files and Segmentations")
        exit_app = file_menu.Append(wx.ID_EXIT, "Exit")
        
        menubar.Append(file_menu, "&File")
        
        # Mesh Generation Menu 
        mesh_menu = wx.Menu() 
        generate_point_cloud = mesh_menu.Append(wx.ID_ANY, "Generate Point Cloud")
        generate_tetra_mesh = mesh_menu.Append(wx.ID_ANY, "Generate Tetra Mesh")
        clean_tetra_mesh = mesh_menu.Append(wx.ID_ANY, "Clean Tetra Mesh")
        mesh_quality = mesh_menu.Append(wx.ID_ANY, "Extract Mesh Quality")
        
        menubar.Append(mesh_menu, "Model Processing")
        
        # Set the menu bar to the frame
        self.SetMenuBar(menubar)
        
        # Bind menu items to corresponding methods
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_dicom_series(1), load_image_view_1)
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_segmentation(1), load_segmentation_view_1)
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_dicom_series(2), load_image_view_2)
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_segmentation(2), load_segmentation_view_2)
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_dicom_series(3), load_image_view_3)
        self.Bind(wx.EVT_MENU, lambda event: self.current_page.load_segmentation(3), load_segmentation_view_3)
        
        self.Bind(wx.EVT_MENU, self.on_generate_point_cloud, generate_point_cloud)
        self.Bind(wx.EVT_MENU, self.on_generate_tetra_mesh, generate_tetra_mesh)
        self.Bind(wx.EVT_MENU, self.on_clean_tetra_mesh, clean_tetra_mesh)
        self.Bind(wx.EVT_MENU, self.on_extract_mesh_quality, mesh_quality)
        
        # Example bindings for save and clear operations (you can replace with actual methods)
        self.Bind(wx.EVT_MENU, self.save_point_cloud, save_point_cloud)
        self.Bind(wx.EVT_MENU, self.save_tetra_cloud, save_tetra_cloud)
        self.Bind(wx.EVT_MENU, self.clear_all_files, clear_all_files)
        self.Bind(wx.EVT_MENU, self.close_program, exit_app)


        
    def add_page(self, name, page_class): 
        page = page_class(self.page_container)
        self.page_sizer.Add(page, 1, wx.EXPAND)
        self.pages[name] = page
        page.Hide()  # Initially hide all pages
    
    def show_page(self, name): 
        if self.current_page:
            self.current_page.Hide()
        self.current_page = self.pages[name]
        self.current_page.Show()
        self.page_container.Layout()
        
    def save_point_cloud(self, event):
        wx.MessageBox("Save Point Cloud functionality not yet implemented.", "Info", wx.OK | wx.ICON_INFORMATION)

    def save_tetra_cloud(self, event):
        wx.MessageBox("Save Tetra Cloud functionality not yet implemented.", "Info", wx.OK | wx.ICON_INFORMATION)
        
    def clear_all_files(self, event):
        # Clear all data
        self.dicom_data = {}
        self.segmentation_data = {}
        wx.MessageBox("All files and segmentations have been cleared.", "Info", wx.OK | wx.ICON_INFORMATION)
    
    def close_program(self, event):
        self.Close()


    def getMaskOverlay(self, masks, volume):
        """Generate an overlay of the segmentation mask on the volume."""
        
        # Ensure volume is a NumPy array of a numeric type
        if not isinstance(volume, np.ndarray):
            volume = np.array(volume)
        if volume.dtype.kind not in {'i', 'u', 'f'}:  # Check if dtype is integer, unsigned, or float
            raise ValueError("Volume data must be numeric. Ensure it is a NumPy array with an appropriate dtype.")
        
        Burned = np.zeros((np.size(masks, 0), np.size(masks, 1), 3, np.size(masks, 2)), dtype=np.uint8)
        Volume2 = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX)
        Volume2 = cv2.convertScaleAbs(Volume2)
    
        Volume_color = np.zeros(Burned.shape, dtype=np.uint8)
        Volume_color[:, :, 0, :] = Volume2
        Volume_color[:, :, 1, :] = Volume2
        Volume_color[:, :, 2, :] = Volume2
    
        num_segs = int(np.max(masks))
        masks_color = np.zeros(Volume_color.shape, dtype=np.uint8)
    
        for ii in range(1, num_segs + 1):  # Segments start from 1 (assuming 0 is background)
            dummy_masks = np.zeros(masks.shape, dtype=np.uint8)
            dummy_masks[masks == ii] = 1
    
            if ii == 1:
                masks_color[:, :, 0, :] += dummy_masks * 255
                masks_color[:, :, 1, :] += dummy_masks * 255
            elif ii == 2:
                masks_color[:, :, 2, :] += dummy_masks * 255
            elif ii == 3:
                masks_color[:, :, 0, :] += dummy_masks * 255
            elif ii == 4:
                masks_color[:, :, 1, :] += dummy_masks * 255
            else:
                rand_r = random.randint(0, 255)
                rand_g = random.randint(0, 255)
                rand_b = random.randint(0, 255)
    
                masks_color[:, :, 0, :] += dummy_masks * rand_r
                masks_color[:, :, 1, :] += dummy_masks * rand_g
                masks_color[:, :, 2, :] += dummy_masks * rand_b
    
        for jj in range(masks.shape[2]):
            if np.sum(masks[:, :, jj]) > 0:
                Burned[:, :, :, jj] = cv2.addWeighted(Volume_color[:, :, :, jj], 0.85, masks_color[:, :, :, jj], 0.15, 0)
            else:
                Burned[:, :, 0, jj] = Volume2[:, :, jj]
                Burned[:, :, 1, jj] = Volume2[:, :, jj]
                Burned[:, :, 2, jj] = Volume2[:, :, jj]
    
        return Burned
    
    def open_point_cloud_options(self, event):
        """Open the Point Cloud Options dialog."""
        dialog = PointCloudOptions(self)
    
        if dialog.ShowModal() == wx.ID_OK:
            # Access the values selected in the dialog
            self.colormap = dialog.colormap
            self.point_size = dialog.point_size
            self.merging_tolerance = dialog.merging_tolerance
    
            # Call the generate_point_cloud method with the selected values
            self.generate_point_cloud(
                colormap=self.colormap,
                point_size=self.point_size,
                merging_tolerance=self.merging_tolerance
            )
        dialog.Destroy()
    
    def on_generate_point_cloud(self, event):
        """Handle Generate Point Cloud menu option."""
        try:
            coords1 = self.dicom_data[1]["coords"] if 1 in self.dicom_data else None
            masks1 = self.segmentation_data[1] if 1 in self.segmentation_data else None
            coords2 = self.dicom_data[2]["coords"] if 2 in self.dicom_data else None
            masks2 = self.segmentation_data[2] if 2 in self.segmentation_data else None
            coords3 = self.dicom_data[3]["coords"] if 3 in self.dicom_data else None
            masks3 = self.segmentation_data[3] if 3 in self.segmentation_data else None
    
            dialog = PointCloudOptions(self)
            if dialog.ShowModal() == wx.ID_OK:
                colormap_name = dialog.colormap
                point_size = dialog.point_size
                merging_tolerance = dialog.merging_tolerance
                whichmask = dialog.whichmask
    
                # Generate the point cloud
                self.last_point_cloud = generate_point_cloud(
                    coords1=coords1, masks1=masks1,
                    coords2=coords2, masks2=masks2,
                    coords3=coords3, masks3=masks3,
                    whichmask=whichmask,
                    tol=merging_tolerance,
                    colormap_name=colormap_name,
                    point_size=point_size
                )
            dialog.Destroy()
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
    
    def on_generate_tetra_mesh(self, event):
        """Handle Generate Tetra Mesh menu option."""
        try:
            if not hasattr(self, 'last_point_cloud') or self.last_point_cloud is None:
                raise ValueError("Point cloud has not been generated. Please generate a point cloud first.")
    
            tetra_mesh = generate_tetra_mesh(self.last_point_cloud)
            self.last_tetra_mesh = tetra_mesh
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
    
    def on_clean_tetra_mesh(self, event):
        """Handle Clean Tetra Mesh menu option."""
        try:
            # Ensure a tetrahedral mesh exists
            if not hasattr(self, 'last_tetra_mesh') or self.last_tetra_mesh is None:
                raise ValueError("Tetrahedral mesh has not been generated. Please generate a tetrahedral mesh first.")
    
            # Open the cleaning options dialog
            dialog = CleanTetraMeshOptions(self)
            if dialog.ShowModal() == wx.ID_OK:
                # Retrieve user inputs
                subdivisions = dialog.subdivisions
                poisson_iterations = dialog.poisson_iterations
                clean_tolerance = dialog.clean_tolerance
                quality_threshold = dialog.quality_threshold
    
                # Clean the tetrahedral mesh
                cleaned_mesh = clean_tetra_mesh(
                    self.last_tetra_mesh,
                    subdivisions=subdivisions,
                    poisson_iterations=poisson_iterations,
                    clean_tolerance=clean_tolerance,
                    quality_threshold=quality_threshold
                )
                self.last_cleaned_mesh = cleaned_mesh
    
                # Visualize the cleaned mesh
                plotter = pv.Plotter()
                plotter.add_mesh(cleaned_mesh, scalars="Elevation", cmap="coolwarm", show_edges=True, opacity=0.8)
                plotter.set_background("black")
                plotter.show(title="Cleaned Tetrahedral Mesh Visualization", interactive_update=True)
            dialog.Destroy()
        except Exception as e:
            wx.MessageBox(f"Failed to clean tetrahedral mesh:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)


    def on_extract_mesh_quality(self, event):
        """Handle Extract Mesh Quality menu option."""
        try:
            # Ensure a cleaned tetrahedral mesh exists
            if not hasattr(self, 'last_cleaned_mesh') or self.last_cleaned_mesh is None:
                raise ValueError("Cleaned tetrahedral mesh is not available. Please clean the mesh first.")
    
            # Call the get_cell_quality function
            quality_mesh = get_cell_quality(self.last_cleaned_mesh)
    
            # Optionally store the quality mesh for future use
            self.last_quality_mesh = quality_mesh
            wx.MessageBox("Mesh quality analysis complete.", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Failed to extract mesh quality:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)


    



   

class StartPage(wx.ScrolledWindow): 
    def __init__(self, parent):
        super().__init__(parent) 
        
        # Enable scrolling
        self.SetScrollRate(10, 10)
        
        # Main vertical sizer
        self.sizer = wx.BoxSizer(wx.VERTICAL) 
        
        # Set background color to black
        self.SetBackgroundColour(wx.Colour(0, 0, 0))  # Black background
        
        # Title
        title = wx.StaticText(self, label="Cardiac Meshalyzer")
        font = wx.Font(24, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        title.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
        self.sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        
        # Placeholder for image
        self.image_path = "mesh_intro_pic.png"
        self.bitmap = wx.StaticBitmap(self)
        self.sizer.Add(self.bitmap, 1, wx.ALL | wx.EXPAND, 10)
        
        # Warning message
        warning = "Warning: Use Cardiac Meshalyzer at your own risk."
        warn_label = wx.StaticText(self, label=warning)
        font_warn = wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL)
        warn_label.SetFont(font_warn)
        warn_label.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
        self.sizer.Add(warn_label, 0, wx.ALL | wx.CENTER, 10)
        
        # Button sizer
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        close_btn = wx.Button(self, label="Close")
        continue_btn = wx.Button(self, label="Continue")
        btn_sizer.Add(close_btn, 0, wx.ALL, 10)  # Close button on the left
        btn_sizer.Add(continue_btn, 0, wx.ALL, 10)  # Continue button on the right
        self.sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)

        # Set button background and text color
        close_btn.SetBackgroundColour(wx.Colour(50, 50, 50))  # Dark grey buttons
        close_btn.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
        continue_btn.SetBackgroundColour(wx.Colour(50, 50, 50))  # Dark grey buttons
        continue_btn.SetForegroundColour(wx.Colour(255, 255, 255))  # White text

        # Bind events
        close_btn.Bind(wx.EVT_BUTTON, self.close_program)
        continue_btn.Bind(wx.EVT_BUTTON, self.open_home_page)
        self.Bind(wx.EVT_SIZE, self.on_resize)  # Bind resize event

        self.SetSizer(self.sizer)
        self.load_image()  # Load the initial image
        self.FitInside()  # Ensure scrolling works with the content size
        
    def load_image(self):
        try:
            img = wx.Image(self.image_path, wx.BITMAP_TYPE_PNG)
            # Scale the image to the current window size
            width, height = self.GetSize()
            img = img.Scale(width - 40, int((width - 40) * 0.5), wx.IMAGE_QUALITY_HIGH)
            self.bitmap.SetBitmap(wx.Bitmap(img))
        except Exception as e:
            # If the image fails to load
            self.bitmap.SetLabel("Image not available.")
        
    def on_resize(self, event):
        self.load_image()  # Reload the image to scale with the window
        event.Skip()  # Ensure other resize events are processed
    
    def open_home_page(self, event): 
        top_level_window = wx.GetTopLevelParent(self)
        top_level_window.show_page("Home Page")
    
    def close_program(self, event):
        wx.GetTopLevelParent(self).Close()
        
        
        
class HomePage(wx.Panel): 
    def __init__(self, parent):
        super().__init__(parent)

        # Set dark background
        self.SetBackgroundColour(wx.Colour(30, 30, 30))  # Dark grey background

        # Main sizer for the page
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left panel for buttons
        button_panel = wx.Panel(self)
        button_sizer = wx.GridSizer(rows=3, cols=2, hgap=10, vgap=10)  # 3 rows x 2 columns grid

        # Buttons for loading image views and segmentations
        load_image_1_btn = wx.Button(button_panel, label="Load Image View 1")
        load_segmentation_1_btn = wx.Button(button_panel, label="Load Segmentation View 1")
        load_image_2_btn = wx.Button(button_panel, label="Load Image View 2")
        load_segmentation_2_btn = wx.Button(button_panel, label="Load Segmentation View 2")
        load_image_3_btn = wx.Button(button_panel, label="Load Image View 3")
        load_segmentation_3_btn = wx.Button(button_panel, label="Load Segmentation View 3")

        # Add buttons to the grid sizer
        button_sizer.Add(load_image_1_btn, 0, wx.EXPAND)
        button_sizer.Add(load_segmentation_1_btn, 0, wx.EXPAND)
        button_sizer.Add(load_image_2_btn, 0, wx.EXPAND)
        button_sizer.Add(load_segmentation_2_btn, 0, wx.EXPAND)
        button_sizer.Add(load_image_3_btn, 0, wx.EXPAND)
        button_sizer.Add(load_segmentation_3_btn, 0, wx.EXPAND)

        button_panel.SetSizer(button_sizer)
        button_panel.SetBackgroundColour(wx.Colour(30, 30, 30))  # Match dark background

        # Right panel for view set buttons and image viewer
        right_panel = wx.Panel(self)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Horizontal sizer for View Set buttons
        view_set_sizer = wx.BoxSizer(wx.HORIZONTAL)
        view_set_1_btn = wx.Button(right_panel, label="View Seg 1")
        view_set_2_btn = wx.Button(right_panel, label="View Seg 2")
        view_set_3_btn = wx.Button(right_panel, label="View Seg 3")

        # Set button styles for dark theme
        for btn in [view_set_1_btn, view_set_2_btn, view_set_3_btn]:
            btn.SetBackgroundColour(wx.Colour(50, 50, 50))  # Dark grey buttons
            btn.SetForegroundColour(wx.Colour(255, 255, 255))  # White text

        # Add View Set buttons to the horizontal sizer
        view_set_sizer.Add(view_set_1_btn, 1, wx.ALL | wx.EXPAND, 5)
        view_set_sizer.Add(view_set_2_btn, 1, wx.ALL | wx.EXPAND, 5)
        view_set_sizer.Add(view_set_3_btn, 1, wx.ALL | wx.EXPAND, 5)

        # Image display and slider
        self.image_display = wx.StaticBitmap(right_panel)
        self.slider = wx.Slider(right_panel, minValue=0, maxValue=1, value=0, style=wx.SL_HORIZONTAL)

        # Add components to the right sizer
        right_sizer.Add(view_set_sizer, 0, wx.ALL | wx.EXPAND, 5)
        right_sizer.Add(self.image_display, 1, wx.ALL | wx.EXPAND, 5)
        right_sizer.Add(self.slider, 0, wx.ALL | wx.EXPAND, 5)

        right_panel.SetSizer(right_sizer)
        right_panel.SetBackgroundColour(wx.Colour(30, 30, 30))  # Match dark background

        # Bind events to the buttons
        load_image_1_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_dicom_series(1))
        load_segmentation_1_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_segmentation(1))
        load_image_2_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_dicom_series(2))
        load_segmentation_2_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_segmentation(2))
        load_image_3_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_dicom_series(3))
        load_segmentation_3_btn.Bind(wx.EVT_BUTTON, lambda event: self.load_segmentation(3))

        view_set_1_btn.Bind(wx.EVT_BUTTON, lambda event: self.view_set(1))
        view_set_2_btn.Bind(wx.EVT_BUTTON, lambda event: self.view_set(2))
        view_set_3_btn.Bind(wx.EVT_BUTTON, lambda event: self.view_set(3))
        self.slider.Bind(wx.EVT_SLIDER, self.update_image)

        # Add the left and right panels to the main sizer
        main_sizer.Add(button_panel, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(right_panel, 1, wx.ALL | wx.EXPAND, 10)

        self.SetSizer(main_sizer)

        # Instance variables for image stack
        self.current_image_stack = None

    def load_dicom_series(self, view_num):
        """Handler for loading a DICOM series."""
        with wx.DirDialog(self, f"Select Folder for Image View {view_num}", style=wx.DD_DEFAULT_STYLE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                folder_path = dlg.GetPath()
                top_level_window = wx.GetTopLevelParent(self)
                volume, coords, out_images = top_level_window.vinayDicomSeries(folder_path)
    
                # Store results with Volume explicitly included
                top_level_window.dicom_data[view_num] = {
                    "volume": volume,
                    "coords": coords,
                    "out_images": out_images
                }
                wx.MessageBox(f"Successfully loaded DICOM series for View {view_num}.", "Data Loaded", wx.OK | wx.ICON_INFORMATION)


    # In load_segmentation method
    def load_segmentation(self, view_num):
        """Handler for loading a segmentation."""
        with wx.FileDialog(
            self,
            message=f"Select Segmentation File for View {view_num}",
            wildcard="All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                mask_path = dlg.GetPath()
                top_level_window = wx.GetTopLevelParent(self)
                masks = top_level_window.get_masks(mask_path)
    
                # Store results in shared dictionary
                top_level_window.segmentation_data[view_num] = masks
                wx.MessageBox(f"Segmentation for View {view_num} loaded successfully.", "Segmentation Loaded", wx.OK | wx.ICON_INFORMATION)


    def view_set(self, view_num):
        """Handler for viewing a set."""
        top_level_window = wx.GetTopLevelParent(self)
        # In view_set method
        try:
            # Check if images and masks are loaded
            dicom_data = top_level_window.dicom_data.get(view_num)
            segmentation_data = top_level_window.segmentation_data.get(view_num)
        
            if dicom_data is None:
                raise KeyError(f"Images for View {view_num} are not loaded.")
            if segmentation_data is None or segmentation_data.size == 0:
                raise KeyError(f"Segmentation for View {view_num} is not loaded or is empty.")
        
            # Extract the Volume component
            volume = dicom_data["volume"]
            masks = segmentation_data
        
            # Call the getMaskOverlay function
            image_stack = top_level_window.getMaskOverlay(masks, volume)
        
            # Store the image stack and configure the slider
            self.current_image_stack = image_stack
            self.slider.SetMax(image_stack.shape[3] - 1)
            self.update_image(None)  # Display the first slice initially
        except KeyError as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)





    def update_image(self, event):
        """Update the displayed image based on the slider value and fit it to the screen."""
        if self.current_image_stack is not None:
            slice_idx = self.slider.GetValue()
    
            # Extract the correct slice
            image_slice = self.current_image_stack[:, :, :, slice_idx]  # Shape (rows, cols, 3)
            
            # Optionally rotate the image to fit better (90 degrees clockwise)
            image_slice = np.rot90(image_slice, k=-1)  # Rotate 90 degrees clockwise
    
            # Flatten and ensure the correct data type
            image_data = image_slice.ravel().astype(np.uint8)
    
            # Resize the image to fill the screen while maintaining aspect ratio
            display_width, display_height = self.image_display.GetSize()
            original_height, original_width = image_slice.shape[:2]
            
            # Calculate scaling factor to maintain aspect ratio
            scale_factor = min(display_width / original_width, display_height / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
    
            # Convert the resized image to wx.Image
            resized_image = cv2.resize(image_slice, (new_width, new_height), interpolation=cv2.INTER_AREA)
            flat_resized_image = resized_image.ravel().astype(np.uint8)
    
            image = wx.Image(new_width, new_height)
            image.SetData(flat_resized_image)
    
            # Update the StaticBitmap
            self.image_display.SetBitmap(wx.Bitmap(image))


class PointCloudOptions(wx.Dialog):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, title="Point Cloud Generation Options", *args, **kwargs)

        self.SetSize((400, 400))  # Set the size of the dialog
        self.Center()  # Center the dialog on the screen

        # Create a vertical sizer
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Colormap Selection
        colormap_label = wx.StaticText(self, label="Colormap:")
        sizer.Add(colormap_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        colormap_choices = plt.colormaps()  # Get all colormap options from matplotlib
        self.colormap_combo = wx.ComboBox(self, choices=colormap_choices, style=wx.CB_READONLY)
        sizer.Add(self.colormap_combo, 0, wx.ALL | wx.EXPAND, 5)

        # Point Size Slider
        point_size_label = wx.StaticText(self, label="Point Size:")
        sizer.Add(point_size_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        point_size_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.point_size_slider = wx.Slider(self, minValue=3, maxValue=20, value=3, style=wx.SL_HORIZONTAL)
        self.point_size_value = wx.StaticText(self, label=str(self.point_size_slider.GetValue()))
        point_size_sizer.Add(self.point_size_slider, 1, wx.ALL | wx.EXPAND, 5)
        point_size_sizer.Add(self.point_size_value, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(point_size_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Merging Tolerance Slider
        merging_tolerance_label = wx.StaticText(self, label="Merging Tolerance:")
        sizer.Add(merging_tolerance_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        merging_tolerance_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.merging_tolerance_slider = wx.Slider(
            self, minValue=0, maxValue=500, value=0, style=wx.SL_HORIZONTAL
        )
        self.merging_tolerance_value = wx.StaticText(self, label=f"{self.merging_tolerance_slider.GetValue() / 100:.2f}")
        merging_tolerance_sizer.Add(self.merging_tolerance_slider, 1, wx.ALL | wx.EXPAND, 5)
        merging_tolerance_sizer.Add(self.merging_tolerance_value, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(merging_tolerance_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Mask Selection
        whichmask_label = wx.StaticText(self, label="Which Mask (Enter Number):")
        sizer.Add(whichmask_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.whichmask_text = wx.TextCtrl(self, value="1")
        sizer.Add(self.whichmask_text, 0, wx.ALL | wx.EXPAND, 5)

        # Generate Point Cloud Button
        self.generate_button = wx.Button(self, label="Generate Point Cloud")
        sizer.Add(self.generate_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)

        # Set the sizer
        self.SetSizer(sizer)

        # Bind events
        self.generate_button.Bind(wx.EVT_BUTTON, self.on_generate_point_cloud)
        self.point_size_slider.Bind(wx.EVT_SLIDER, self.update_point_size_value)
        self.merging_tolerance_slider.Bind(wx.EVT_SLIDER, self.update_merging_tolerance_value)

        # Variables to hold widget values
        self.colormap = None
        self.point_size = 3
        self.merging_tolerance = 0.0
        self.whichmask = 1

    def update_point_size_value(self, event):
        """Update the displayed value for the point size slider."""
        self.point_size_value.SetLabel(str(self.point_size_slider.GetValue()))

    def update_merging_tolerance_value(self, event):
        """Update the displayed value for the merging tolerance slider."""
        self.merging_tolerance_value.SetLabel(f"{self.merging_tolerance_slider.GetValue() / 100:.2f}")

    def on_generate_point_cloud(self, event):
        """Retrieve widget values and close the dialog."""
        self.colormap = self.colormap_combo.GetValue()
        self.point_size = self.point_size_slider.GetValue()
        self.merging_tolerance = self.merging_tolerance_slider.GetValue() / 100.0  # Convert to 0-5 range

        try:
            self.whichmask = int(self.whichmask_text.GetValue())
        except ValueError:
            wx.MessageBox("Invalid mask number. Please enter an integer.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.EndModal(wx.ID_OK)

class CleanTetraMeshOptions(wx.Dialog):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, title="Clean Tetra Mesh Options", *args, **kwargs)

        self.SetSize((400, 400))  # Set the dialog size
        self.Center()  # Center the dialog on the screen

        # Create a vertical sizer
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Number of Subdivisions
        subdivisions_label = wx.StaticText(self, label="Subdivisions:")
        sizer.Add(subdivisions_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.subdivisions_text = wx.TextCtrl(self, value="2")
        sizer.Add(self.subdivisions_text, 0, wx.ALL | wx.EXPAND, 5)

        # Poisson Iterations
        poisson_iterations_label = wx.StaticText(self, label="Poisson Iterations:")
        sizer.Add(poisson_iterations_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.poisson_iterations_text = wx.TextCtrl(self, value="10")
        sizer.Add(self.poisson_iterations_text, 0, wx.ALL | wx.EXPAND, 5)

        # Cleaning Tolerance
        clean_tolerance_label = wx.StaticText(self, label="Cleaning Tolerance:")
        sizer.Add(clean_tolerance_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.clean_tolerance_text = wx.TextCtrl(self, value="0.001")
        sizer.Add(self.clean_tolerance_text, 0, wx.ALL | wx.EXPAND, 5)

        # Quality Threshold
        quality_threshold_label = wx.StaticText(self, label="Quality Threshold:")
        sizer.Add(quality_threshold_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.quality_threshold_text = wx.TextCtrl(self, value="1e-5")
        sizer.Add(self.quality_threshold_text, 0, wx.ALL | wx.EXPAND, 5)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(self, label="OK")
        cancel_button = wx.Button(self, label="Cancel")
        button_sizer.Add(ok_button, 0, wx.ALL, 10)
        button_sizer.Add(cancel_button, 0, wx.ALL, 10)
        sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL)

        self.SetSizer(sizer)

        # Bind buttons
        ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)

        # Initialize variables to hold user input
        self.subdivisions = 2
        self.poisson_iterations = 10
        self.clean_tolerance = 0.001
        self.quality_threshold = 1e-5

    def on_ok(self, event):
        """Save the user inputs and close the dialog."""
        try:
            self.subdivisions = int(self.subdivisions_text.GetValue())
            self.poisson_iterations = int(self.poisson_iterations_text.GetValue())
            self.clean_tolerance = float(self.clean_tolerance_text.GetValue())
            self.quality_threshold = float(self.quality_threshold_text.GetValue())
            self.EndModal(wx.ID_OK)
        except ValueError:
            wx.MessageBox("Please enter valid numerical values.", "Error", wx.OK | wx.ICON_ERROR)

    def on_cancel(self, event):
        """Close the dialog without saving."""
        self.EndModal(wx.ID_CANCEL)

#%% 


if __name__ == "__main__": 
    app = wx.App(False)
    frame = CardiacMeshalyzer(None)
    app.MainLoop() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        