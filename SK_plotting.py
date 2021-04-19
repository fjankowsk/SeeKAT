#!/usr/bin/env python
#Tiaan Bezuidenhout, 2020. For inquiries: bezmc93@gmail.com
#NB: REQUIRES Python 2

'''
Plotting tools.
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units  as u

import SK_utils as ut
import SK_coordinates as co


def plot_known(w,known_coords):
	'''
	Makes a cross on the localisation plot where a known source is located.
	w must be a WCS object pre-set with SK_coordinates.buildWCS
	'''

	known_coords = known_coords.split(',')

	try:
		known_coords = [float(known_coords[0]),float(known_coords[1])]
	except:
		known_coords = SkyCoord(known_coords[0],known_coords[1], frame='icrs', unit=(u.hourangle, u.deg))
		known_coords = [known_coords.ra.deg,known_coords.dec.deg]


	known_px = w.all_world2pix([known_coords],1)
	
	plt.scatter(known_px[0,0],known_px[0,1],c='red',marker='x',s=200,zorder=999)

def make_ticks(array_width,array_height,w,fineness):
	'''
	Adds ticks and labels in sky coordinates to the plot.
	'''

	ticks = ut.getTicks(array_width,array_height,w,fineness)
	labels = co.pix2deg(ticks,w)
	ra_deg= np.around(labels[:,0],4)
	dec_deg = np.around(labels[:,1],4)
	plt.xticks(ticks[0], ra_deg,rotation=40,fontsize=8)
	plt.yticks(ticks[1], dec_deg,fontsize=8)

def likelihoodPlot(ax,likelihood):
	'''
	Creates the localisation plot
	'''

	plt.imshow(likelihood,origin='lower',cmap='inferno')

	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Localisation probability')
	plt.scatter(np.where(likelihood==np.amax(likelihood))[1],np.where(likelihood==np.amax(likelihood))[0],marker='v',c='cyan')
#	plt.contour(likelihood,levels=[0.9],zorder=800,colors='cyan')
#	plt.contour(likelihood,levels=[0.68],zorder=800,colors='lime')
#	plt.contour(likelihood,levels=[0.9],zorder=800,colors='cyan')
#	plt.contour(likelihood,levels=[0.32],zorder=800,colors='lime')
        #plt.contour(likelihood,levels=[1-np.std(likelihood)],zorder=800,colors='cyan')
	#plt.contour(likelihood,levels=[1-2*np.std(likelihood)],zorder=800,colors='lime')

	plt.xlabel('RA ($^\circ$)')
	plt.ylabel('Dec ($^\circ$)')
	
	#ax.set_xlim(450,670)
	#ax.set_ylim(450,670)
        
        likelihood /= likelihood.sum()

        ## Calculating the interval values in 2D
        likelihood_flat_sorted = np.sort(likelihood, axis=None)
        likelihood_flat_sorted_cumsum = np.cumsum(likelihood_flat_sorted)
        ind_1sigma = np.nonzero(likelihood_flat_sorted_cumsum > (1-0.6827))[0][0]
        ind_2sigma = np.nonzero(likelihood_flat_sorted_cumsum > (1-0.9545))[0][0]
        ind_3sigma = np.nonzero(likelihood_flat_sorted_cumsum > (1-0.9973))[0][0]
        val_1sigma = likelihood_flat_sorted[ind_1sigma]
        val_2sigma = likelihood_flat_sorted[ind_2sigma]
        val_3sigma = likelihood_flat_sorted[ind_3sigma]
        max_loc = np.where(likelihood==np.amax(likelihood))
        ## When displaying a 2D array, the last index is the last axis, thus we need to flip things here.
        max_loc = [max_loc[1],max_loc[0]]
        
        ## Calculating the interval values in 1D for y
        likelihood_1D = likelihood.sum(1)
        likelihood_1D_flat_sorted = np.sort(likelihood_1D, axis=None)
        likelihood_1D_flat_sorted_cumsum = np.cumsum(likelihood_1D_flat_sorted)
        val_1D_1sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.6827))[0][0]]
        val_1D_2sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.9545))[0][0]]
        val_1D_3sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.9973))[0][0]]
        ind_1D_1sigma = np.nonzero(likelihood_1D > val_1D_1sigma)[0]
        ind_1D_2sigma = np.nonzero(likelihood_1D > val_1D_2sigma)[0]
        ind_1D_3sigma = np.nonzero(likelihood_1D > val_1D_3sigma)[0]
        val_1sigma_y = [ind_1D_1sigma[0], ind_1D_1sigma[-1]]
        val_2sigma_y = [ind_1D_2sigma[0], ind_1D_2sigma[-1]]
        val_3sigma_y = [ind_1D_3sigma[0], ind_1D_3sigma[-1]]
        
        ## Calculating the interval values in 1D for x
        likelihood_1D = likelihood.sum(0)
        likelihood_1D_flat_sorted = np.sort(likelihood_1D, axis=None)
        likelihood_1D_flat_sorted_cumsum = np.cumsum(likelihood_1D_flat_sorted)
        val_1D_1sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.6827))[0][0]]
        val_1D_2sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.9545))[0][0]]
        val_1D_3sigma = likelihood_1D_flat_sorted[np.nonzero(likelihood_1D_flat_sorted_cumsum > (1-0.9973))[0][0]]
        ind_1D_1sigma = np.nonzero(likelihood_1D > val_1D_1sigma)[0]
        ind_1D_2sigma = np.nonzero(likelihood_1D > val_1D_2sigma)[0]
        ind_1D_3sigma = np.nonzero(likelihood_1D > val_1D_3sigma)[0]
        val_1sigma_x = [ind_1D_1sigma[0], ind_1D_1sigma[-1]]
        val_2sigma_x = [ind_1D_2sigma[0], ind_1D_2sigma[-1]]
        val_3sigma_x = [ind_1D_3sigma[0], ind_1D_3sigma[-1]]
        
        plt.contour(likelihood,levels=[val_1sigma],zorder=800,colors='cyan')
        plt.contour(likelihood,levels=[val_2sigma],zorder=800,colors='lime')

        return max_loc, val_1sigma_x, val_2sigma_x, val_3sigma_x, val_1sigma_y, val_2sigma_y, val_3sigma_y
