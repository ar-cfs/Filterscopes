import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from time import time
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

def Grade_Stack(stackL, wvl, atomicS, prevTok):
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

    #spec = np.array([s.split(' ')[0] for s in atomicS])
    out1 = np.zeros((wvl.shape[-1],3),dtype = np.float16)


    #all the same for each row, so we can just use the first row
    cspec = np.array([s.split(' ')[0] for s in atomicS[:,0]])

    #find the indexes of the speces in the stackL
    idx = [np.searchsorted(cspec, substack) for substack in stackL]

    for i in range(len(idx)):
        cwvl = np.take(wvl,idx[i],axis =0)
        #cas = np.take(atomicS,idx[i],axis =0)
        cpt = np.take(prevTok,idx[i],axis =0)

        t = time()
        print(f'Grading stack {i+1}/{len(idx)}...')

        wvlScore = Grade_Wvl(cwvl)
        ptScore = Grade_PrevTok(cpt)
        uvScore = Grade_UV(cwvl)

        print(f'Stack {i+1} graded in {time()-t:.2f} seconds')

        out1[:,0] += wvlScore
        out1[:,1] += ptScore
        out1[:,2] += uvScore
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
    return out1

def Plot_Stack(scores,stackL,wvl,atomicS,prevTok,wvlW = 1, uvSW = 1, prevTokW = 1):
    """
    Plot the scores for each stack
    Inputs:
    - scores: nparray- the scores for each stack
    - stackL: list of lists- the stacks to plot
    - wvl: nparray- the wavelengths for each stack
    - atomicS: nparray- the atomic states for each stack
    - prevTok: nparray- the previous tokamak usage for each stack
    """

    scores[:,0] *= wvlW
    scores[:,1] *= prevTokW
    scores[:,2] *= uvSW
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

        #set the title with the total score and weighting
        total_score = weightS[i]
        wvl_score = scores[i,0]
        prevTok_score = scores[i,1]
        uv_score = scores[i,2]
        

        a[0].set_title(f' Total Score: {total_score:.2f}, Wvl Score: {wvl_score:.2f}, UV Score: {uv_score:.2f}, Prev Tok Score: {prevTok_score:.2f}')

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

    height = 1

    for i in range(wvl.shape[0]):
        cclr = wave_to_rgb(wvl[i])
        ax.vlines(wvl[i], 0,height,color=cclr)
        ax.text(wvl[i]+12, 0.8, f'{spec[i]} {ion[i]}', rotation=0, ha='center', va='bottom', fontsize=11, color=cclr)
    ax.set_xlabel('Wavelength (nm)')


if __name__ == '__main__':
    filename = 'lines.xlsx'
    saveF = 'TestStack.npz'
    
    stackL =[ ['B','He','O'],['N','C','B','W']]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              [ 'He', 'Ne', 'Ar', 'Kr'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],\
                ]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],]
    #wvl, spec, ion, prevTok= Load_data(filename)
    #outWvl, outAS, outPT = Filter_Stacks(stackL,wvl, spec, ion, prevTok)
    #Save_Stack('TestStack', outWvl,outAS,outPT)

    outWvl,outAS,outPT = Load_Stack(saveF)

    scores = Grade_Stack(stackL, outWvl, outAS, outPT)
    #Save_GradedStack(saveF, outWvl,outAS,outPT,scores)

    outWvl, outAS, outPT, scores = Load_GradedStack(saveF)

    Plot_Stack(scores, stackL, outWvl, outAS, outPT,prevTokW = 0.5,uvSW = 2)

    #Plot_lines(wvl, spec, ion)
    
