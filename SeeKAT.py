#!/usr/bin/env python
#Tiaan Bezuidenhout, 2020. For inquiries: bezmc93@gmail.com
#NB: REQUIRES Python 2

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

import SK_utils as ut
import SK_coordinates as co
import SK_plotting as Splot

from scipy.stats import norm

def ratio_likelihood(z,S1,S2,dS1=1.0,dS2=1.0):

        a = np.sqrt(z ** 2 / dS1 ** 2 + 1.0 / dS2 ** 2)
        b = S1 / dS1 ** 2 * z + S2 / dS2 ** 2
        c = S1 ** 2 / dS1 ** 2 + S2 ** 2 / dS2 ** 2
        d = np.exp((b ** 2 - c * a ** 2) / (2 * a ** 2))

        like = (1.0/(np.sqrt(2.0 * np.pi) * dS1 * dS2) * b * d / a ** 3
                * (norm.cdf(b/a,scale=1.0) - norm.cdf(-b/a,scale=1.0))
                + 1.0/(a ** 2 * np.pi * dS1 * dS2) * np.exp(-c/2))

        return like

def parseOptions(parser):
	'''Options:
	-f	Input file with each line a different CB detection.
		Should have 3 columns: RA (h:m:s), Dec (d:m:s), S/N
	-p	PSF of a CB in fits format
	--o	Fractional sensitivity level at which CBs are tiled to overlap
	--r	Resolution of PSF in units of arcseconds per pixel
	--n	Number of beams to consider when creating overlap contours. Will
		pick the specified number of beams with the highest S/N values.
	--s Draws known coordinates onto the plot for comparison.    
	'''

	parser.add_argument('-f', dest='file', 
				nargs = 1, 
				type = str, 
				help="Detections file",
				required=True)
	parser.add_argument('-c', dest='config', 
				nargs = 1, 
				type = str, 
				help="Configuration (json) file",
				required=False)
	parser.add_argument('-p',dest='psf',
				nargs=1,
				type=str,
				help="PSF file",
				required=True)
	parser.add_argument('--o', dest='overlap',
				type = float,
				help = "Fractional sensitivity level at which the coherent beams overlap",
				default = 0.25,
				required = False)
	parser.add_argument('--r', dest='res',
				nargs = 1,
				type=float,
				help="Distance in arcseconds represented by one pixel of the PSF",
				default = 1,
				required = True)
	parser.add_argument('--n',dest='npairs',
				nargs = 1,
				type = int,
				default = [1000000])
	parser.add_argument('--s', dest='source',
				nargs = 1,
				type=str,
				help="Draws given coordinate location (degrees) on localisation plot",
				required = False)


	options= parser.parse_args()

	return options


def make_plot(array_height,array_width,c,psf_ar,options,data):
    
    sum_threshold = ut.get_best_pairs(data,options.npairs[0])   # Only beam pairs with S/N summing to above this
                                                                # number will be used for localisation

    full_ar = np.zeros((array_height,array_width))

    loglikelihood = np.zeros((array_height,array_width))

    for i in range(0,len(c)):
        #print i
        beam_ar = np.zeros((array_height,array_width))

        dec_start = int(c.dec.px[i])-int(psf_ar.shape[1])/2
        dec_end = int(c.dec.px[i])+int(psf_ar.shape[1])/2
        ra_start = int(c.ra.px[i])-int(psf_ar.shape[0])/2
        ra_end = int(c.ra.px[i])+int(psf_ar.shape[0])/2

        beam_ar[dec_start : dec_end,ra_start : ra_end] = psf_ar
        plt.contour(beam_ar,levels=[options.overlap],colors='white',linewidths=0.5,linestyles='dashed') # shows beam sizes

        full_ar = np.maximum(full_ar,beam_ar)

        for j in range(0,len(c)):
            stdout.write("\rComputing localisation curves for beam %d vs %d/%d..." % (j+1,i+1,len(c)))
            stdout.flush()
            if i<j and data["SN"][i]+data["SN"][j] >= sum_threshold:
                #f, ax = plt.subplots()

                plt.scatter(c.ra.px,c.dec.px,color='white',s=0.2)
                #plt.scatter(c.ra.px[i],c.dec.px[i],color='magenta')
                #plt.scatter(c.ra.px[j],c.dec.px[j],color='magenta')
                comparison_ar = np.zeros((array_height,array_width))

                dec_start = int(c.dec.px[j])-int(psf_ar.shape[1])/2
                dec_end = int(c.dec.px[j])+int(psf_ar.shape[1])/2
                ra_start = int(c.ra.px[j])-int(psf_ar.shape[0])/2
                ra_end = int(c.ra.px[j])+int(psf_ar.shape[0])/2

                comparison_ar[dec_start : dec_end,
                    ra_start : ra_end] = psf_ar

                plt.contour(comparison_ar,levels=[options.overlap],colors='white',linewidths=0.5,linestyles='dashed')
                plt.contour(beam_ar,levels=[options.overlap],colors='white',linewidths=0.5,linestyles='dashed')

                beam_snr = data["SN"][i]
                comparison_snr = data["SN"][j]

                loglikelihood = localise(beam_snr,comparison_snr,beam_ar,comparison_ar,loglikelihood)
                #Splot.make_ticks(array_width,array_height,w,fineness=40)
                #Splot.likelihoodPlot(ax,likelihood)
                #plt.show()
                #plt.savefig('Frame%d_%d' % (i,j),dpi=300)
    #plt.imshow(full_ar,origin='lower',cmap='inferno')
    #plt.show()
    #likelihood /= np.amax(likelihood)
    
    return loglikelihood


def localise(beam_snr,comparison_snr,beam_ar,comparison_ar,loglikelihood):
	'''
	Plots contours where the ratio of the S/N detected in each 
	beam to the highest-S/N detection matches the ratio of 
	those beams' PSFs. 1-sigma errors are also drawn.
	'''

        ratio_ar = np.divide(beam_ar,comparison_ar)

        like = ratio_likelihood(ratio_ar,beam_snr,comparison_snr)
        logl = np.nan_to_num(like)

        loglikelihood += np.log(logl)

        return loglikelihood

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    options = parseOptions(parser)
    
    data,c,boresight = ut.readCoords(options)
    
    psf_ar = ut.readPSF(options.psf[0])
    
    c,w,array_width,array_height = co.deg2pix(c,psf_ar,boresight,options.res[0])
    
    f, ax = plt.subplots()
    
    if options.source:
        Splot.plot_known(w,options.source[0])

    Splot.make_ticks(array_width,array_height,w,fineness=50)
    
    loglikelihood = make_plot(array_height,array_width,c,psf_ar,options,data)
    likelihood = np.exp(loglikelihood - np.nanmax(loglikelihood))
    Splot.likelihoodPlot(ax,likelihood)
    max_deg = []
    max_loc = np.where(likelihood==np.amax(likelihood))
    
    if len(max_loc) == 2:
        max_loc = (max_loc[1],max_loc[0])
        ut.printCoords(max_loc,w)
    else:
        print 'Multiple equally possible locations'
    plt.show()
