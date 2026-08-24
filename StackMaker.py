import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from time import time
import drjit as dr
import h5py
import os
from scipy.signal import find_peaks


colormap = [[6, 1, 31],[12, 0, 40],[14, 0, 51], [16, 1, 60],
 [17, 1, 76], [23, 0, 90], [26, 1, 105], [28, 0, 119], [28, 0, 136], 
[34, 0, 151],[36, 1, 165], [37, 0, 176], [37, 1, 187], [36, 0, 194],
[37, 0, 202], [34, 0, 209], [31, 0, 217], [28, 1, 220], [25, 0, 224],
[18, 1, 227], [16, 0, 229], [14, 0, 233], [10, 0, 237], [9, 0, 237],
[7, 0, 240], [3, 0, 242], [0, 0, 244], [0, 0, 244], [2, 5, 244], 
[1, 8, 244], [0, 13, 242], [0, 18, 242], [2, 22, 239],
[0, 28, 236], [0, 33, 236], [0, 37, 232], [0, 44, 229], [2, 49, 227],
[0, 55, 220], [0, 60, 218], [0, 66, 214], [1, 73, 209], [0, 77, 205],
[0, 84, 200], [0, 91, 194], [0, 96, 193], [0, 101, 189], [0, 106, 182],
[0, 111, 177], [1, 118, 172], [0, 120, 165], [0, 122, 159], 
[0, 128, 153], [1, 131, 147], [1, 132, 140], [1, 135, 134], 
[0, 140, 131], [0, 145, 126], [0, 148, 124], [0, 152, 122], 
[0, 158, 118], [1, 162, 118], [1, 168, 116], [0, 172, 114],
[0, 178, 113],[0, 182, 112], [0, 186, 111], [1, 188, 109], 
[2, 191, 107], [0, 194, 107], [1, 195, 108], [0, 198, 101], 
[1, 200, 99], [0, 204, 96], [1, 209, 97], [2, 211, 94], [1, 217, 90],
[0, 220, 88], [0, 225, 81], [1, 228, 77], [1, 231, 71], [1, 232, 68], 
[0, 230, 60], [0, 230, 52], [0, 230, 43], [0, 230, 33], [0, 228, 21],
[0, 228, 11], [2, 229, 0], [16, 229, 0], [28, 229, 0], [40, 230, 0], 
[56, 232, 0], [72, 232, 2], [84, 230, 1],
[98, 231, 0],[111, 230, 0],[124, 230, 0], [137, 230, 1], [151, 228, 0],
[162, 227, 0], [173, 229, 0], [186, 227, 0], [198, 224, 1], 
[211, 226, 0], [221, 221, 0], [227, 216, 0], [230, 210, 1], 
[237, 201, 1], [240, 193, 1],[242, 184, 0], [245, 173, 0], 
[248, 165, 1], [250, 155, 0], [251, 145, 0], [252, 136, 1], 
[254, 126, 1], [255, 115, 0], [255, 104, 3], [254, 95, 1], [255, 83, 1],
[255, 72, 2], [255, 61, 0], [253, 49, 0], [255, 39, 2],
[253, 28, 0], [255, 17, 4], [255, 8, 1], [254, 2, 1], [254, 0, 10],
[255, 0, 14], [255, 0, 18], [251, 0, 24], [250, 0, 30], [250, 0, 30], 
[248, 0, 35], [246, 0, 41], [246, 0, 41], [242, 0, 40], [242, 0, 40],
[240, 0, 45], [237, 0, 46], [233, 0, 45], [230, 1, 44], [226, 0, 42], 
[222, 0, 41], [218, 0, 39], [214, 0, 38], [206, 0, 36], [200, 1, 34], 
[195, 0, 32], [189, 0, 30], [185, 0, 31], [177, 0, 28], [169, 0, 26],
[162, 0, 24], [152, 0, 23],[144, 1, 21], [136, 1, 18], [128, 1, 20], 
[121, 0, 19], [111, 0, 16], [104, 0, 14], [96, 0, 12], [88, 1, 10], 
[83, 0, 12], [73, 0, 9], [67, 0, 9], [62, 1, 9], [57, 0, 7], [51, 0, 7],
[46, 0, 5], [42, 0, 4], [39, 0, 5], [33, 1, 4], [30, 0, 4],[25, 0, 3], 
[25, 0, 3], [22, 0, 2],[21, 0, 1], [16, 0, 0], [15, 1, 1], [14, 0, 0], 
[12, 0, 0], [9, 0, 1], [9, 0, 1], [8, 0, 0]]


def wave_to_rgb(wavlgth):
    """Input : a float describing a wavelength in nanometer
    Output : a numpy array giving the rgb values (between 0 and 1) 
    associated with the colour percieved at this wavelength """
    a = np.linspace(400,700,len(colormap))
    colorindex = min(range(len(a)), key=lambda i: abs(a[i]-wavlgth))
    col = colormap[colorindex]
    return np.asarray(col)/255

def LowHighCent(wvl,intens ):
    """
    Find the low, high, and center wavelengths of a line given its intensity profile.
    Currently finds the lowest and highest non-zero intensity points as the low and high.
    Assumes wvl and intens are sorted in ascending order of wavelength.
    Inputs:
    - wvl: nparray - the wavelength array
    - intens: nparray - the intensity array
    Outputs:
    - low: float - the low wavelength of the line
    - high: float - the high wavelength of the line
    - center: float - the center wavelength of the line
    """
    #cind = find_peaks(intens,prominence = max(intens)/20,distance = 500)
    cind = find_peaks(intens,prominence = max(intens)/30,distance = 500)
    centers = wvl[cind[0]]
    low = np.zeros_like(centers)
    high = np.zeros_like(centers)


    maxW = wvl[-1]-wvl[0]
    minW = np.diff(centers)/2

    if len(minW)==0:
        minW = maxW
    else:
        minW = minW.min()


    for i, center in enumerate(centers):
        peak_intensity = intens[cind[0][i]]
        threshhold = peak_intensity * 0.01  # 1% of peak value

        #find low and high wavelengths by finding first threshold intensity points
        #within minW of center

        mask = np.logical_and(wvl>center-minW, wvl<center+minW)

        subW = wvl[mask & (intens > threshhold)]
        low[i] = subW[0]
        high[i] = subW[-1]

    print('Number of peaks found:', len(centers))
    return low, high, centers

def Load_H5_Zeeman(folder='broad',wvlD = 0.01,plot = False):
    """Process h5 files containing zeeman split lines to create a excel file
    which contains the low, high, and center wavelengths of each line. Must be run once
    before using Prep_Zeeman.
    Inputs:
    - folder: str - the folder containing the h5 files
    - wvlD: float - the spacing for the wavelength grid in nm, default is 0.1 nm
    Outputs:
    - wvl: nparray - the wavelength grid for the lines
    - ovrlp: nparray - normalized summed spectra of all the lines in folder
    """

    wvl = np.arange(300,900, wvlD)
    out = np.zeros_like(wvl)

    

    files = [f for f in os.listdir(folder) if f.endswith('.h5')]


    fOut = []
    centers = []
    lows = []
    highs = []


    for j,file in enumerate(files):
        print(file)
        f = h5py.File(os.path.join(folder, file), 'r')

        sig = f['signal'][:]
        cwvl = f['wave_air'][:]


        #sort by wavelength
        sInd = np.argsort(cwvl)
        cwvl = cwvl[sInd]
        sig = sig[sInd]



        if np.any(np.isnan(sig)):
            print('NaN values found in file:', file)
            continue

        #all-zero signal (e.g. unsupported 1P-1S term pairs) would give 0/0 = NaN
        #in the rescale below and poison the summed overlap array
        if sig.max() == sig.min():
            print('Flat/all-zero signal found in file, skipping:', file)
            continue



        #find the lowest and highest non-zero intensity points
        low, high, center = LowHighCent(cwvl, sig)

        for i in range(len(low)):
            fOut.append(file.split('.h5')[0])
            centers.append(center[i])
            lows.append(low[i])
            highs.append(high[i])

        
        if plot:
            fig,a = plt.subplots(1,1)
            a.set_title(file)
            a.plot(cwvl,sig)

            for i in range(len(low)):
                a.axvline(low[i],color='red',ls='--')
                a.axvline(high[i],color='red',ls='--')
                a.axvline(center[i],color='blue',ls='--')
            a.set_xlabel('Wavelength (nm)')
            a.set_ylabel('Intensity (a.u.)')
            plt.show()




        #rescale to between 0 and 1
        sig = (sig - sig.min())/(sig.max()-sig.min())


        #interpolate sig onto wvl and add to out
        out += np.interp(wvl, cwvl, sig, left=0.0, right=0.0)

    #save the center, low and high wavelengths in a csv file
    with open(os.path.join(folder, 'wavelengths.csv'), 'w', newline='') as csvfile:
        fieldnames = ['file', 'low', 'high', 'center']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(lows)):
            writer.writerow({'file': fOut[i], 'low': lows[i], 'high': highs[i], 'center': centers[i]})

    if plot:
        f,a = plt.subplots(1,1)
        a.plot(wvl, out, color='black', linewidth=2)

        a.set_xlabel('Wavelength (nm)')
        a.set_ylabel('Overlap (-n)')
        a.set_title('Zeeman Overlap')
        plt.show()
    return wvl, out

def Process_Zeeman(center, low,high, wvlD = 0.1,plot = False):
    """
    Calculates how much lines overlap due to Zeeman splitting
    returns a wavelength grid an associated overlap array which
    which indicates how much the lines overlap
    where 0 is no overlap and -n is the amount of overlap
    i.e. if n lines perfectly overlap, -n is returned
    a flat top function is used to indicate the overlap
    Inputs:
    - center: nparray - the center wavelengths of the lines
    - low: nparray - lower wavelength "width" of the line in center
    - high: nparray - higher wavelength "width" of the line in center
    - wvlD: float - the spacing for the wavelength grid in nm, default is 0.1 nm
    Outputs:
    - wvl: nparray - the wavelength grid for the lines
    """
    pad = np.max(high-low)/2

    wvl = np.arange(center.min() - pad, center.max() + pad, wvlD)
    overlap = np.zeros_like(wvl)

    

    for i in range(len(center)):
        #find the indices of the wavelengths which are within the width of the line
        mask = np.logical_and(wvl>=low[i], wvl<=high[i])

        #set the overlap to -1 for the indices which are within the width
        overlap[mask] -= 1
    
    if plot:
        f,a = plt.subplots(1,1)
        a.plot(wvl, overlap, color='black', linewidth=2)

        #add vertical lines colored by the wavelength
        for i in range(len(center)):
            a.axvline(center[i], color=wave_to_rgb(center[i]), linewidth=2,alpha = 0.1,ls = '--')
        a.set_xlabel('Wavelength (nm)')
        a.set_ylabel('Overlap (-n)')
        a.set_title('Zeeman Overlap')
        plt.show()
    return wvl, overlap

def Prep_Zeeman(filename, wvlD = 0.1, plot = False):
    #load csv file
    
    zeemanD = pd.read_csv(filename)
    wvlOvlp,ovrlp = Process_Zeeman(zeemanD['center'], zeemanD['low'], zeemanD['high'], wvlD = wvlD,plot = plot)

    return wvlOvlp,ovrlp

def Load_xlsx(filename):
    data = pd.read_excel(filename, sheet_name='Sheet1',header = 0,usecols = [1,2,3,12,13,14,16])
    wvl = np.array(data['wave']).astype(float)
    spec = np.array(data['element']).astype(str)
    ion = np.array(data['charge']).astype(str)
    species = spec + ' ' + ion
    pec = np.array(data['PEC_ind']).astype(int)
    uInd = np.array(data['up_ind']).astype(int)
    lInd = np.array(data['low_ind']).astype(int)

    

    #stack pec,uInd, lInd to a n,3 array
    adas = np.stack((pec, uInd, lInd), axis=1)
    """
    #find the rows where all entries are -1
    mask = np.all(adas == -1, axis=1)

    #remove those rows from wvl, species, and prevTok
    wvl = wvl[~mask]
    species = species[~mask]
    adas = adas[~mask]
    spec = spec[~mask]
    ion = ion[~mask]
    """
    
    width = np.array(data['Zee_width (nm)']).astype(float)

    ###
    #place filler while Curt is working on getting zeeman widths
    mask = np.isnan(width)
    tempW = np.unique(width[~mask])

    #randomly assign widths from the non-nan values to the nan values
    width[mask] = np.random.choice(tempW, size=np.sum(mask), replace=True)
    ###

    prevTok = np.full_like(species, False, dtype=bool)

    return wvl, spec, species, ion, prevTok,width


def Load_data(filename):
    """
    Load data from an xlsx file
    Data should have lambda, Species and atomic state, and if it was used in a tokamak
    Inputs:
    - filename: str- the name of the file to load
    Outputs:
    - wvl: nparray- the wavelengths of the lines
    - species: nparray- the species of the lines
    - ion: nparray- the ionization state of the lines
    - prevTok: nparray- boolean array indicating if the line was used in a tokamak
    """

    """
    #For previous google sheet 
    data = pd.read_excel(filename, sheet_name='Sheet1',header = 0,usecols = [0,1,5])
    #convert it to numpy arrays
    wvl = np.array(data['lambda']).astype(float)
    species = np.array(data['atomic state']).astype(str)
    prevTok = np.array(data['Used on Tok Machine']).astype(str)

    #remove the rows where either wvl is NaN pr species is NaN
    mask = ~np.logical_or(species=='nan',np.isnan(wvl))

    wvl = wvl[mask]
    species = species[mask]
    prevTok = prevTok[mask]

    #parse species to ge the symbol and ionization state
    spec = np.array([s.split(' ')[0] for s in species])
    ion = np.array([s.split(' ')[1] for s in species])
    print('unique lines:', wvl.shape[0])
    """

    wvl, spec, species, ion, prevTok,_ = Load_xlsx(filename)



    print('unique lines:', wvl.shape[0])

    #this is super sloppy
    wvlOut = []
    specOut = []
    ionOut = []
    prevTokOut = []




    #for lines within one nm of each other, downselect to one of them
    for csp in np.unique(spec):
        print(csp)
        maskTop = spec==csp
        cwvl = wvl[maskTop]

        sInd = np.argsort(cwvl)
        cwvl = cwvl[sInd]


        cion = ion[maskTop][sInd]
        cprevTok = prevTok[maskTop][sInd]

        

        gI = np.full_like(cwvl,False)
        for i,lamb in enumerate(cwvl):
            #if statement so that we don't drop the first line within 1 nm of the group
            if gI[i]:
                continue

            diff = cwvl[i+1:]-lamb
            mask = diff<1
            
            gI[i+1:] = mask

        #now remove the lines which are within 1 nm of each other
        gI = ~gI.astype(bool)

        for i in range(len(gI)):
            if gI[i]:
                wvlOut.append(cwvl[i])
                specOut.append(csp)
                ionOut.append(cion[i])
                prevTokOut.append(cprevTok[i])




    print('unique lines after downselect:', len(wvlOut))
    wvl = np.asarray(wvlOut).flatten()
    spec = np.asarray(specOut).flatten()
    ion = np.asarray(ionOut).flatten()
    prevTok = np.asarray(prevTokOut).flatten()



    #convert prevtok to a boolean array
    prevOut = np.full(len(prevTok), False, dtype=bool)
    prevOut[prevTok == 'Y'] = True

    return wvl, spec, ion, prevOut

def Pattern(arr,repeats,tiles):
    """
    Create a pattern of the array with the given number of repeats and tiles
    Effectively used for creating a for loop over the array
    Inputs:
    - arr-nparray- the array to repeat and tile
    - repeats- int-  the number of times to repeat the array
    - tiles -  int- the number of times to tile the array
    Outputs:
    - arr-nparray- the repeated and tiled array
    """
    if repeats !=0:
        arr = np.repeat(arr, repeats)
    if tiles != 0:
        arr = np.tile(arr, (1,tiles))
    return arr



def Filter_Stacks(stackL,wvl,spec,ion,prevTok):
    """
    Creates a list of all the possible lines (wavelengths) for each element
    specified in the stack.
    Inputs:
    - stackL: list of lists - shape (x,y-) each sublist contains the elements which will be in a "stack", i.e. measured simultaneously
     e.g. [['O', 'N', 'C', 'B', 'He'], ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C']]
    - wvl: nparray-(n) the wavelengths of the possible lines to observe
    - spec: nparray-(n) the species of the possible lines to observe
    - ion: nparray- (n)the ionization state of the possible lines to observe
    - prevTok: nparray (n)- boolean array indicating if the line was used in a tokamak
    Outputs:
    - outWvl: nparray- the wavelengths of the lines for each possible stack, shape (unique species, tot wavelength combinations)
    where x is the number of species and tot is the total number of combinations of lines
    - outAS: nparray- the species and ionization state of the lines for each possible stack, shape (unique species, tot wavelength combinations)
    - outPT: nparray- boolean array indicating if the line was used in a tokamak for each possible stack, shape (unique species, tot wavelength combinations)
    """

    uspec = []
    for i in range(len(stackL)):
        for j in range(len(stackL[i])):
            uspec.append(stackL[i][j])
    uspec = np.array(uspec)
    uspec = np.unique(uspec)

    num =np.zeros(uspec.shape[0], dtype=int)
    #calculate the number of lines for each species
    for i, s in enumerate(uspec):
        num[i] = np.sum(s==spec)
    
    #multiply the number of lines for each species

    tot = np.prod(num)

    outWvl = np.zeros((uspec.shape[0],tot),dtype = np.float16)
    outAS = np.zeros((uspec.shape[0],tot), dtype='<U3')
    outPT = np.zeros((uspec.shape[0],tot), dtype=bool)
    #fill out with all the combinations of the spec lines

    t  = time()
    print('Generating combinations...')
    print(f'Total combinations: {tot}')

    for i, s in enumerate(uspec):
        #this is basically nested for loops, but we use the Pattern function to create the combinations
        #this is needed since each species can have a different number of lines

        repI = i+1
        repeats = np.prod(num[repI:])

        tileI = i
        tiles = np.prod(num[:tileI])
        if tileI == 0:
            tiles = 0
        
        mask = spec == s
        wvlS = wvl[mask]
        ionS = s+' '+ion[mask]
        prevTokS = prevTok[mask]



        #repeat the wvlS for the number of repeats
        outWvl[i,:] = Pattern(wvlS,repeats, tiles)
        outAS[i,:] = Pattern(ionS,repeats, tiles)
        outPT[i,:] = Pattern(prevTokS,repeats, tiles)


    print(f'Combinations generated in {time()-t:.2f} seconds')

    return outWvl, outAS, outPT

def Grade_Zeeman(wvl, overWvl, Overlap, plot=False, chunkN = 20_000_000):
    """
    Memory-efficient version of Grade_Zeeman.
    Processes the combination axis in chunks of chunkN so no single
    host-pinned/GPU allocation exceeds a few hundred MB (a 6-species x 98M
    combination stack needs a 4 GB pinned transfer in one shot, which the
    driver refuses).
    """
    import drjit as dr

    ncomb = wvl.shape[-1]
    out = np.empty(ncomb, dtype=np.float32)

    for s in range(0, ncomb, chunkN):
        e = min(s + chunkN, ncomb)
        cwvl = np.ascontiguousarray(wvl[:, s:e], dtype=np.float32)

        idx = dr.full(dr.cuda.TensorXf,0,cwvl.shape)
        smllD = dr.full(dr.cuda.TensorXf, dr.inf,cwvl.shape)  # Use dr.full for GPU compatibility

        wvlT = dr.cuda.TensorXf(cwvl) #GPU array

        # Iterate over each wavelength in overWvl
        #this list is much smaller than the total combinations of stack wavelengths
        #even though it can still be pretty large
        for i, ow in enumerate(overWvl):

            diff = dr.abs(wvlT - ow)#find the closest overlap wavelength to wvl of lines

            # Find where the current difference is smaller than the previous minimum
            mask = diff < smllD
            smllD[mask] = diff[mask]

            # Update the output indices where the current difference is smaller
            idx[mask] = i

        #convert back to numpy array
        idxArr = np.array(idx,dtype=int)
        #sums the overlap values for each possible stack
        out[s:e] = np.take(Overlap, idxArr).sum(0)+ wvl.shape[0]  #add wvl.shape[0] to make 0 no overlap and -n overlap of n lines

        del idx, smllD, wvlT, diff, mask
        dr.flush_malloc_cache()  #release cached GPU/pinned memory before the next chunk
        print(f'chunk {e}/{ncomb} done')

    return out

def Grade_Wvl(wvl,clip = 30):
    """
    Grade the wavelength based on the spacing between lines
    Captures abilitiy to buy a dichoric filter to split the lines in wvl
    returns a score for each possible stack between 0 and 1
    Inputs:
    - wvl: nparray - (unique species, wavelength combinations) - the wavelengths of the lines
    - clip: int - the maximum spacing between lines to be considered good, default is 30 nm
    Outputs:
    - out: nparray - (wavelength combinations) - the score for each stack, where 1 is perfect and 0 is bad
    """

    #sort wvl 
    sWvl = np.sort(wvl,axis = 0)
    """

    mask = np.diff(sWvl,axis = 0)> clip
    #med = np.median(np.diff(sWvl,axis = 0),axis = 0)/clip #any spacing greater than 100 nm is fine

    #sum the mask to get the number of lines which are spaced more than 30 nm apart
    out = np.sum(mask,axis = 0)/(sWvl.shape[0]-1)  #percentage of lines spaced more than 30 nm apart
    """

    #alternatively, create a linear function which punishes lines below 30 nm
    out = np.clip(np.diff(sWvl,axis = 0),None,clip) / clip  #clip the spacing to be below 30 nm and normalize by 30 nm
    out = out.sum(0)/(sWvl.shape[0]-1)

    return out #return the minimum of the median spacing or 1 (100% score)

def Grade_PrevTok(prevTok):
    """
    Grade the previous tokamak usage
    Returns a score for each line in the stack
    The score is based on the previous tokamak usage
    Inputs:
    - prevTok: nparray - (unique species, wavelength combinations) - boolean array indicating if the line was used in a tokamak
    Outputs:
    - out: nparray - (wavelength combinations) - the score for each stack, where 1 is perfect and 0 is bad
    """

    return np.sum(prevTok,axis=0)/prevTok.shape[0]  #percentage of lines used in a tokamak

def Grade_UV(wvl,cutoff = 450,lwvl = 350):
    """
    Grade the wavelength based on the UV range
    punishes lines below cutoff linearly from 0 to 1
    wavelengths below lwvl are set to 0
    Inputs:
    - wvl: nparray - (unique species, wavelength combinations) - the wavelengths of the lines
    - cutoff: int - the cutoff wavelength for the UV range, default is 450 nm
    - lwvl: int - the lower wavelength limit for the UV range, default is 350 nm
    Outputs:
    - out: nparray - (wavelength combinations) - the score for each stack, where 1 is perfect and 0 is bad
    """
    #line from 0 to 1 from lwvl to cutoff
    cwvl = np.clip(wvl, lwvl, None)  #clip the wavelengths to be above lwvl
    cwvl = np.clip(wvl, None, cutoff)  #clip the wavelengths to be below cutoff
    cwvl = (cwvl - lwvl) / (cutoff - lwvl)  #normalize the wavelengths to be between 0 and 1

    #now sum the scores and normalize by the number of lines

    out = cwvl.sum(0)/ wvl.shape[0]  #percentage of lines in the UV range
    #np.sum(wvl > cutoff,axis = 0)/wvl.shape[0] 
    return out  

def Grade_Stack(stackL, wvl, atomicS, prevTok,filename,lowMem = True):
    """
    Grade the stack based on the wavelength, atomic state and previous tokamak usage
    Returns a score for each line in the stack
    The score is based on the wavelength, atomic state and previous tokamak usage
    Inputs:
    - stackL: list of lists  each sublist contains the elements which will be in a "stack", i.e. measured simultaneously
     e.g. [['O', 'N', 'C', 'B', 'He'], ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C']]
    - wvl: nparray - (unique species, wavelength combinations) - the wavelengths of the lines
    - atomicS: nparray - (unique species, wavelength combinations) - the species and ionization state of the lines
    - prevTok: nparray - (unique species, wavelength combinations) - boolean array indicating if the line was used in a tokamak
    Outputs:
    - out: nparray - (wavelength combinations, 3) - the scores for each stack, where the first column is the wavelength score,
     the second column is the previous tokamak usage score and the third column is the UV score
    """

    #calcualte three metrics
    #1) median wavelength range
    #2) previous tokamak usage
    #3) number of lines below a certain wavelength

    
    #setup the overlap and wavelength arrays
    wvlOvr,ovrlp = Prep_Zeeman(filename, wvlD = 0.25, plot = True)

    #spec = np.array([s.split(' ')[0] for s in atomicS])

    out1 = np.zeros((wvl.shape[-1],4),dtype = np.float16)


    #all the same for each row, so we can just use the first row
    cspec = np.array([s.split(' ')[0] for s in atomicS[:,0]])

    #find the indexes of the speces in the stackL
    idx = [np.searchsorted(cspec, substack) for substack in stackL]

    
    bestScore = -np.inf
    bestStack = []
    bestScores = (0,0,0,0)

    for i in range(len(idx)):
        cwvl = np.take(wvl,idx[i],axis =0)
        #cas = np.take(atomicS,idx[i],axis =0)
        cpt = np.take(prevTok,idx[i],axis =0)

        t = time()
        print(f'Grading stack {i+1}/{len(idx)}...')


        #out1[:,0] += Grade_Wvl(cwvl)
        print('wvl done')
        #out1[:,1] += Grade_PrevTok(cpt)
        print('Prev Tok done')
        #out1[:,2] += Grade_UV(cwvl)
        print('UV done')
        out1[:,3] += Grade_Zeeman(cwvl,wvlOvr,ovrlp)
        print('Zeeman done')

        print(f'Stack {i+1} graded in {time()-t:.2f} seconds')

        #cScore = (wvlScore + ptScore + uvScore + zScore)  #negative score for minimization
        """
        if cScore > bestScore:
            bestScore = cScore
            bestStack = stackL[i]
            bestScores = (wvlScore, ptScore, uvScore, zScore)
            print(f'New best stack: {bestStack} with scores Wvl: {wvlScore:.2f}, PT: {ptScore:.2f}, UV: {uvScore:.2f}, Z: {zScore:.2f}, Total: {-bestScore:.2f}')
        
        print(f'Stack {i+1} graded in {time()-t:.2f} seconds')
        if not lowMem:
            out1[:,0] += wvlScore
            out1[:,1] += ptScore
            out1[:,2] += uvScore
            out1[:,3] += zScore
        """
    """
    t = time()

    print('Grading stacks with different wavelengths...')
    out = np.zeros((wvl.shape[-1],3))
    for cwvl,cas, cpt in zip(wvl.T, atomicS.T, prevTok.T):
        

        #now grade each stack individually
        for i,substack in enumerate(stackL):
            subidx = idx[i]
            #get the wavelengths for the substack
            subwvl = cwvl[subidx]
            subas = cas[subidx]
            subpt = cpt[subidx]

            #grade the stack
            wvlScore = Grade_Wvl(subwvl)
            ptScore = Grade_PrevTok(subpt)
            uvScore = Grade_UV(subwvl)

            out[totInd] += [wvlScore, ptScore, uvScore]

            #print(f'Stack: {substack}, Wavelength Score: {wvlScore:.2f}, Previous Tokamak Score: {ptScore:.2f}, UV Score: {uvScore:.2f}')

        totInd += 1

        if totInd % 10000 == 0:
            print(f'Processed {totInd} stacks in {time()-t:.2f} seconds')
    print(f'Graded {totInd} stacks in {time()-t:.2f} seconds')
    """
    #plot the best stacks

    return out1

def Plot_Stack(scores,stackL,wvl,atomicS,prevTok,wvlW = 1, uvSW = 1, prevTokW = 1,zW = 1,zFilename = 'sparc_line_ids_widths_v0.xlsx'):
    """
    Plot the scores for each stack
    Inputs:
    - scores: nparray- the scores for each stack
    - stackL: list of lists- the stacks to plot
    - wvl: nparray- the wavelengths for each stack
    - atomicS: nparray- the atomic states for each stack
    - prevTok: nparray- the previous tokamak usage for each stack
    """
    #wvlOvl,ovrlp = Prep_Zeeman(zFilename, wvlD = 0.5, plot = True)
    wvlOvl,ovrlp = Load_H5_Zeeman(folder='broad_all_fixed',wvlD = 0.01,plot = True)
    scores[:,0] *= wvlW
    scores[:,1] *= prevTokW
    scores[:,2] *= uvSW
    scores[:,3] *= zW
    weightS = np.sum(scores, axis=1)

    #sort the list and order from highest to lowest
    sorted_indices = np.argsort(weightS)[::-1]

    cspec = np.array([s.split(' ')[0] for s in atomicS[:,0]])

    #find the indexes of the speces in the stackL
    idx = [np.searchsorted(cspec, substack) for substack in stackL]


    for i in sorted_indices:
        f,a = plt.subplots(len(stackL),1,sharex = True)

        subWvl = wvl[:,i]
        subSpec = atomicS[:,i]

        for j in range(len(stackL)):
            cwvl = np.take(subWvl,idx[j])

            cas = np.take(subSpec,idx[j])


            Plot_Lines(cwvl,cas,None,a[j])
            a[j].plot(wvlOvl, ovrlp, color='black', linewidth=1)

        #set the title with the total score and weighting
        total_score = weightS[i]
        wvl_score = scores[i,0]
        prevTok_score = scores[i,1]
        uv_score = scores[i,2]
        z_score = scores[i,3]

        a[0].set_title(f' Total Score: {total_score:.2f}, Wvl Score: {wvl_score:.2f}, UV Score: {uv_score:.2f}, Prev Tok Score: {prevTok_score:.2f}, Z Score: {z_score:.2f}')

        #remove the y labels
        for ax in a:

            ax.set_yticklabels([])
        plt.show()



def Save_Stack(filename, wvl,atomicS,prevTok):
    np.savez(filename, wvl=wvl, atomicS=atomicS, prevTok=prevTok)

def Save_GradedStack(filename, wvl,atomicS,prevTok,scores):
    np.savez(filename, wvl=wvl, atomicS=atomicS, prevTok=prevTok,scores = scores)

def Load_Stack(filename):
    """
    Load a stack from a file
    """
    data = np.load(filename)
    wvl = data['wvl']
    atomicS = data['atomicS']
    prevTok = data['prevTok']
    return wvl, atomicS, prevTok

def Load_GradedStack(filename):
    """
    Load a stack from a file
    """
    data = np.load(filename)
    wvl = data['wvl']
    atomicS = data['atomicS']
    prevTok = data['prevTok']
    scores = data['scores']
    return wvl, atomicS, prevTok,scores

def Plot_Lines(wvl, spec, ion,ax = None):
    """
    plots all the lines as a vertical line at the wavelength
    labels the lines with the species and ionization state with a small offset
    TODO: convert wvl to rgb color based on the wavelength
    """
    clrL = [plt.cm.rainbow(i) for i in np.linspace(0,1,wvl.shape[0])]

    if ax is None:
        f,ax = plt.subplots(1,1)

    if ion is None:
        ion = np.array(['']*spec.shape[0])

    height = 10

    for i in range(wvl.shape[0]):
        cclr = wave_to_rgb(wvl[i])
        ax.vlines(wvl[i], 0,height,color=cclr)
        #ax.vlines(wvl[i], color=cclr)
        ax.text(wvl[i]+12, 0.8, f'{spec[i]} {ion[i]}', rotation=0, ha='center', va='bottom', fontsize=11, color=cclr)
    ax.set_xlabel('Wavelength (nm)')


if __name__ == '__main__':
    filename = 'lines.xlsx'
    filename = 'sparc_line_ids_v1.xlsx'
    filename = 'sparc_line_ids_widths_v0.xlsx'
    zFolder = 'broad_all_fixed'
    zFolder = 'split_ext_field_50_terms_h5'
    zFile = 'broad_all_fixed/wavelengths.csv'
    saveF = 'TestStack_broadfixed.npz'  #new name so results from the old buggy broad/ data are kept
    
    stackL =[ ['B','He','O'],['N','C','B','W']]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              [ 'He', 'Ne', 'Ar', 'Kr'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],\
                ]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],]
    
    stackL = [['O', 'N', 'C', 'B', 'He'],\
            ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al'],\
            ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],]
    #regenerate wavelengths.csv (low/high/center of each Zeeman-split peak)
    #from the fixed term-grouped h5 dataset
    Load_H5_Zeeman(folder=zFolder,wvlD = 0.01,plot = False)

    wvl, spec, ion, prevTok= Load_data(filename)
    outWvl, outAS, outPT = Filter_Stacks(stackL,wvl, spec, ion, prevTok)

    scores = Grade_Stack(stackL, outWvl, outAS, outPT,zFile)
    Save_GradedStack(saveF, outWvl,outAS,outPT,scores)

    #outWvl, outAS, outPT, scores = Load_GradedStack(saveF)

    #Plot_Stack(scores, stackL, outWvl, outAS, outPT,prevTokW = 0.5,uvSW = 2,zW = 1,zFilename = zFile)

    #Plot_lines(wvl, spec, ion)
