This code base is part of a research article currently under revision by eLife. The corresponding data set can be found in https://datadryad.org/stash/dataset/doi:10.5061/dryad.sj3tx9683
A preprint of the article can be found on biorxiv under https://www.biorxiv.org/content/10.1101/2022.06.01.494332v3

The data in dryad contain all the data after the first layer of image analysis in form of numpy arrays. 
Further analysis of this data was performed with python and the code for this is available under https://github.com/MRT9393/ForceTransmissionInDoublets. The code for the first layer of analysis can also be found there but without the data because of large file sizes.

Scripts for all the simulations are also provided here in the corresponding folders.

Running the scripts corresponding to the second layer of analysis will allow you to reproduce the figures in the article. For this, first the "meta_analysis.py" and "meta_analysis_tissues.py" have to be run. 
These scripts will perform numerous calculations on these data and store them in a compact and organized dictionary file.
After this, the different figure scripts can be run which will produce the plots in the figures.

Detailed description of the data
################################
These folders contain all of the experimental data after TFM analysis and were used to produce all the figures:

	AR1to1_doublets_full_stim_short
	AR1to1_singlets_full_stim_short
	AR2to1_doublets_half_stim
	AR1to1_doublets_half_stim
	AR1to2_doublets_half_stim
	AR1to1_singlets_half_stim
	AR1to1_doublets_full_stim_long
	AR1to1_singlets_full_stim_long
	tissues_lefthalf_stim
	tissues_tophalf_stim

The name of the folders are a codified description of the experiment. 
The first identifier describes the biological object that was studied: 
	AR1to1_doublet for example stands for doublets on H shaped micropatterns with a 1to1 aspect ratio. AR1to2_doublets have a 1to2 aspect ratio. AR1to1 singlets are single cells on H shaped micropatterns with 1to1 aspect ratio and so on.
	Tissues stand for small monolayers of a few dozen cells on rectangular micropatterns.
The second identifier describes the type of optogenetic stimulation that was performed. 
	full_stim means, the whole object was stimulated and half_stim means, only the left half of the biological object was stimulated. 
	For the tissues, only half stimulations were performed, but we stimulated either the top or the left half.
	For most experiments, the cells were stimulated with one pulse per minute for 10 minutes (long stimulation). For some experiments, only three pulses with one pulse per minute were performed (short stimulation).
	Whenever not specified, the long stimulation protocol was applied.

For all experiments except for the tissues, four types of measurements were performed and the files in these folders are the result of these. Here are the different measurement types and their corresponding files:
	
	Actin anisotropy measurements:
	*******************************
	actin_anisotropy.npy

	Actin intensity measurements:
	*****************************
	actin_intensity.npy
	actin_intensity_left.npy
	actin_intensity_right.npy
	cortex_intensity.npy
	cortex_intensity_left.npy
	cortex_intensity_right.npy
	SF_intensity.npy
	SF_intensity_left.npy
	SF_intensity_right.npy


	Traction Force Microscopy (TFM) and Monolayer Stress Microscopy (MSM):
	**********************************************************************
	mask.npy
	Dx.npy
	Dy.npy
	Tx.npy
	Ty.npy
	sigma_xx.npy
	sigma_xy.npy
	sigma_yx.npy
	sigma_yy.npy

	Stress fiber tracking:
	**********************
	Xbottom.npy
	Ybottom.npy
	Xright.npy
	Yright.npy
	Xtop.npy
	Ytop.npy
	Xleft.npy
	Yleft.npy

For the tissue experiment, the same data was acquired, except for the stress fiber tracking data. 
For all experiments, actin images and fluorescent bead images were acquired, from which these measurements were obtained. These raw data are too big to be shared publicly but are available upon request.
Some example actin images are provided for a few of the experiments. Unfortunately, all the images could not be included, since they would exceed the data storage limit. 
These example images are the ones found in the figures and are necessary for their reproduction.
For an explanation of how these measurements were performed, please refer to the Materials&Methods section of the accompanying paper.

Following is a short description of the data contained in these numpy arrays. All arrays can have several dimensions: x, y, t and c (for cell)

	actin_anisotropy.npy contains an array with one value per cell (1,c). These values describe the degree of alignment of the actin fibers and their value is between -1 and 1, with -1 corresponding to perfect, 
	vertical alignment and 1 corresping to perfect, horizontal alignment.

	All the actin intensity files contain arrays with one value per cell and per frame (t,c). These values describe the average actin intensity over time in specific regions of the cell. 
	Left and right -> Left and right half of the image.
	Cortex -> Only the inside of the cell, excluding the stress fibers at its borders.
	SF -> Only the outside of the cell, excluding the cortex on the inside.

	All the TFM and MSM data contain one spatial map per cell and per frame (x, y, t, c). 
	The mask is a binary map, describing the position of each cell at each point in time.
	Dx and Dy describe the substrate displacement in x and in y-direction respectiveley.
	Tx and Ty describe the traction force maps in x and in y-direction respectiveley. These values were calculated from the displacement maps.
	sigma_xx and sigma_yy describe the normal stresses inside the cells in x and in y-direction respectiveley. These values were calculated from the traction force maps and the masks. 
	sigma_xy and sigma_yx describe the shear stresses inside the cells in x and in y-direction respectiveley. These values were calculated from the traction force maps and the masks.

	All the stress fiber tracking data contain the coordinates of n equally spaced points on the stress fibers for each point in time (n, t, c).
	X and Y describe the x and y-coordinates respectiveley.
	Bottom, right, top, left describe the position of the stress fiber.


Description of the "analysed data" folder.
##########################################

This folder is the output directory of all "second layer" analysis perfomed on the previously described data. This data is stored as .dat files, which contain nested dictionaries, providing good organization of the data, which can
comfortably be explored and further analysed with python. The provided code should be used to open these files.

Some of this data, mainly the output of the contour model analysis, is already provided (all the .dat files with a CM in their name).
The analysis runtime to produce these files is very long (several hours), but they can be reproduced with "contour_model_analysis.py" from the provided analysis code.

The .csv files contain aggregated output data from these contour fitting analyses for further data processing. Here, the line tension corresponds to the average force in the two free stress fibers (top and bottom).
Sigma_x_CM and sigma_y_CM correspond to the two components of the anisotropic surface tension coming from the actin cortex.
a and b correspond to the diameter of the two axes describing the ellipse that was fitted to the free stress fibers.
RSI describes the relative surface tension increase after photoactivation, i.e. the surface tension after photoactivation normalized by the surface tension before photoactivation.
The "baseline" file contains an average value for these quantities of all time points before photoactivation and the "full_stim" file contains data of these quantities after photoactivation of the whole system.
The "stats" files contain statistical readouts from these data (median, std, etc.)


Description of the "_FEM_simulations" folder
############################################

This folder contains the results of the Finite-Element-Simulations. 
For the two-dimensional FEM simulations the RAW simulation output (xdmf) was processed with ParaView (open source) in order  
to use ParaView's build in function "cell data to point data" (https://vtk.org/doc/nightly/html/classvtkCellDataToPointData.html).
The point data (displacement field maps, stress maps, traction maps etc.) was then exported to csv format. 
Before storing the csv data to .dat files the maps where further interpolated on a regular grid to match the resolution of the experimental maps. 

The subfolder named "strain_energy_doublets" and "strain_energy_singlets" contain .npz files with the simulated time course of the strain energy. 

Description of the "_contour_simulations" folder
################################################

This folder contains the results of the contour-FEM simulations.
The resulting data is stored as .dat files, which contain nested dictionaries of the relevant quantities.
