# Loads TAPAS tellurics

import numpy as np
import matplotlib.pyplot as plt
import glob
import gzip
import pickle
from scipy.interpolate import RegularGridInterpolator

def load_tellurics(files):
    telfil = glob.glob(files) # reading the tellurics
    airmass,ang = np.zeros(np.size(telfil)),np.zeros(np.size(telfil))
    i = 0
    for ff in telfil:
        with gzip.open(ff) as file:
            data = file.readlines()
            airmass[i] = np.float(data[15].strip()[9:])
            ang[i] = np.float(data[14].strip()[4:])
            i+=1
    data = np.loadtxt(telfil[0], skiprows=23)
    tell = np.zeros((np.size(telfil),data.shape[0],data.shape[1]))
    ii = np.argsort(airmass)
    airmass,ang = airmass[ii],ang[ii]
    for i in range(len(airmass)):
        print(telfil[ii[i]])
        buff = np.loadtxt(telfil[ii[i]], skiprows=23)
        tell[i,:] = buff
    print(ang)
    return tell, airmass, ang

tellw, airw, angw = load_tellurics('../../data/tapas/*winter*ipac.gz')
tells, airs, angs = load_tellurics('../../data/tapas/*summer*ipac.gz')

wavew,waves = np.squeeze(tellw[0,:,0]), np.squeeze(tells[0,:,0])
iiw,iis = np.argsort(wavew),np.argsort(waves)
tellw, tells = tellw[:,iiw,:],tells[:,iis,:]
tellwi = RegularGridInterpolator((airw,np.squeeze(tellw[0,:,0])), np.squeeze(tellw[:,:,1]))
