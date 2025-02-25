#!/usr/bin/env python
#Tiaan Bezuidenhout, 2020. For inquiries: bezmc93@gmail.com
#NB: REQUIRES Python 3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
np.seterr(divide='ignore', invalid='ignore')

import SK.utils as ut
import SK.coordinates as co
import SK.plotting as Splot


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
	--scalebar Sets the length of the scale bar on the plot in arcseconds.
	--ticks Sets the spacing of ticks on the localisation plot.
	--clip Sets level below which CB PSF is set equal to zero.
	--zoom Automatically zooms in on the TABs.
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
				help='Number of beams to use',
				default = [1000000])
	parser.add_argument('--s', dest='source',
				nargs = 1,
				type=str,
				help="Draws given coordinate location (degrees) on localisation plot",
				required = False)
	parser.add_argument('--scalebar', dest='sb',
						nargs = 1,
						type = float,
						help = "Length of scale bar on the localisation plot in arcseconds. Set to 0 to omit altogether",
						default = [10],
						required = False)
	parser.add_argument('--ticks', dest='tickspacing',
						nargs = 1,
						type = float,
						help = "Sets the number of pixels between ticks on the localisation plot",
						default = [100],
						required = False)
	parser.add_argument('--clip', dest='clipping',
						nargs = 1,
						type = float,
						help = "Sets values of the PSF below this number to zero. Helps minimise the influence of low-level sidelobes",
						default = [0.08],
						required = False)	
	parser.add_argument('--zoom', dest='autozoom',
						help = "Automatically zooms the localisation plot in on the TABs",
						action = 'store_true')	
	options= parser.parse_args()

	return options

def make_map(array_height,array_width,c,psf_ar,options,data):
	if options.npairs[0] > 2 and options.npairs[0]+1 <= len(c):
		npairs = options.npairs[0]-1
	else:
		npairs = len(c)-1

	full_ar = np.zeros((array_height,array_width))

	loglikelihood = np.zeros((array_height,array_width))

	nit = 1000  # number of iterations for covariance matrix

	sim_ratios = np.zeros((nit,0))
	obs_ratios = np.zeros((npairs,0))

	#resids = np.zeros((nit,0))
	fake_snrs = data["SN"][None,:] + np.random.randn(nit*len(c)).reshape(nit,len(c))

	#make covariance matrix
	beam_snr = data["SN"][0]
	beam_snrs_fake = fake_snrs[:,0]

	for j in range(1,npairs+1):
		comparison_snr = data["SN"][j]
		comparison_snrs_fake = fake_snrs[:,j]
		sim_ratios = np.append(sim_ratios,(comparison_snrs_fake/beam_snrs_fake)[:,None],axis=1)	# NB, ratios must always be comparison (lower SN) / beam (highest SN)
		obs_ratios = np.append(obs_ratios,(comparison_snr/beam_snr))

	C = np.cov(sim_ratios,rowvar=False)

	# make model and get residuals

	psf_ratios = np.zeros((array_height,array_width,npairs))

	cn = 0
	stdout.write("\rAdding beam %d/%d..." % (1,npairs+1))
	stdout.flush()

	beam_ar = np.zeros((array_height,array_width))
	beam_snr = data["SN"][0]  # NB, beams must be sorted by S/N; highest first!

	dec_start = int(np.round(c.dec.px[0]))-int(psf_ar.shape[1]/2)
	dec_end = int(np.round(c.dec.px[0]))+int(psf_ar.shape[1]/2)
	ra_start = int(np.round(c.ra.px[0]))-int(psf_ar.shape[0]/2)
	ra_end = int(np.round(c.ra.px[0])) +int(psf_ar.shape[0]/2)

	beam_ar[dec_start : dec_end,ra_start : ra_end] = psf_ar
	plt.contour(beam_ar,levels=[options.overlap],colors='black',linewidths=0.5,linestyles='dashed') # shows beam sizes

	for j in range(1,npairs+1):
		stdout.write("\rAdding beam %d/%d..." % (j+1,npairs+1))
		stdout.flush()

		plt.scatter(c.ra.px,c.dec.px,color='black',s=0.2)

		comparison_snr = data["SN"][j]
		# print(comparison_snr,"/",beam_snr)

		comparison_ar = np.zeros((array_height,array_width))

		dec_start = int(np.round(c.dec.px[j]))-int(psf_ar.shape[1]/2)
		dec_end = int(np.round(c.dec.px[j]))+int(psf_ar.shape[1]/2)
		ra_start = int(np.round(c.ra.px[j]))-int(psf_ar.shape[0]/2)
		ra_end = int(np.round(c.ra.px[j]))+int(psf_ar.shape[0]/2)
		comparison_ar[dec_start : dec_end, ra_start : ra_end] = psf_ar
                
		plt.contour(comparison_ar,levels=[options.overlap],colors='black',linewidths=0.5)
		plt.contour(beam_ar,levels=[options.overlap],colors='black',linewidths=0.5)

		psf_ratios[:,:,cn] = comparison_ar/beam_ar
		cn+=1

	resids = np.zeros((array_height,array_width,npairs))

	for i in range(0,npairs):
		resids[:,:,i] = obs_ratios[i] - psf_ratios[:,:,i]

	chi2 = np.sum(resids * np.sum(np.linalg.inv(C)[None,None,:,:] * resids[:,:,:,None],axis=2),axis=2)
	chi2[chi2==np.inf] = np.nan
	loglikelihood = -0.5 * chi2

	return loglikelihood


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    options = parseOptions(parser)
    
    data,c,boresight = ut.readCoords(options)

    psf_ar = ut.readPSF(options.psf[0],options.clipping[0])
    
    c,w,array_width,array_height = co.deg2pix(c,psf_ar,boresight,options.res[0])

    f, ax = plt.subplots(figsize=(10,10))


    if options.source:
        Splot.plot_known(w,options.source[0])
    
    loglikelihood = make_map(array_height,array_width,c,psf_ar,options,data)

    Splot.make_ticks(ax,array_width,array_height,w,fineness=options.tickspacing[0])

    print("\nPlotting...")
    Splot.likelihoodPlot(f,ax,loglikelihood,options)

    max_deg = []
    max_loc = np.where(loglikelihood==np.nanmax(loglikelihood))

    if len(max_loc) == 2:
        max_loc = (max_loc[1],max_loc[0])
        ut.printCoords(max_loc,w)
    # else:
        # print('Multiple equally possible locations')
    
    if options.autozoom == True:
        ax.set_xlim(min(c.ra.px) - int(15/options.res[0]) ,max(c.ra.px) + int(15/options.res[0]))
        ax.set_ylim(min(c.dec.px) -int(15/options.res[0]) ,max(c.dec.px) + int(15/options.res[0]))

    #l = np.exp(loglikelihood - np.nanmax(loglikelihood))
    #l /= l.sum()
    #ut.write2fits(w,l)

    plt.savefig(options.file[0]+'.png',dpi=300)
    plt.show()
