import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from time import time
import drjit as dr
import h5py
import os
import re
from fractions import Fraction
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

#---------------------------------------------------------------------------
# Zeeman width estimation
#
# Curt's h5 term files only cover part of the candidate line list. For the rest
# we estimate the Zeeman envelope width from the atomic terms so that
# Grade_Zeeman sees an envelope for every line instead of silently treating an
# uncomputed line as "no overlap". Measured h5 widths always take precedence;
# these estimates only fill gaps (see Merge_Zeeman_Catalog).
#
# References for the physics used below (all standard, textbook results):
#
# [1] Lande g-factor, g = 1 + [J(J+1)+S(S+1)-L(L+1)]/[2J(J+1)]:
#     E. U. Condon and G. H. Shortley, "The Theory of Atomic Spectra",
#     Cambridge University Press (1935), Ch. XVI (Zeeman effect).
#     R. D. Cowan, "The Theory of Atomic Structure and Spectra",
#     University of California Press (1981), Sec. 16-2.
#     C. J. Foot, "Atomic Physics", Oxford University Press (2005), Sec. 5.5.
#
# [2] Anomalous Zeeman component shifts, dnu = (m_u g_u - m_l g_l)*mu_B*B/(hc)
#     with electric-dipole selection rules dm = 0 (pi), +-1 (sigma):
#     Condon and Shortley [1] Ch. XVI; I. I. Sobelman, "Atomic Spectra and
#     Radiative Transitions", 2nd ed., Springer (1992), Ch. 8.
#     Plasma-diagnostics treatment: H. R. Griem, "Principles of Plasma
#     Spectroscopy", Cambridge University Press (1997), Ch. 4.
#     The coefficient mu_B/(hc) is the Lorentz unit, 0.4669 cm^-1/T
#     (NIST CODATA "Bohr magneton in inverse meters per tesla", 46.686 m^-1/T);
#     the constants below reproduce this to 4 significant figures.
#
# [3] lambda^2 scaling: wavenumber-to-wavelength conversion. With nu = 1/lambda,
#     dnu/dlambda = -1/lambda^2, so |dLambda| = lambda^2 |dnu|. Since the Zeeman
#     shift is constant in wavenumber [2], the split in WAVELENGTH grows as
#     lambda^2 - which is why a flat "widest measured width" is not conservative
#     in the near-IR. See e.g. Griem [3] or any spectroscopy text.
#
# NOTE: the two calibration numbers below (C_MAX_JK and the default `safety`
# factor) are NOT from the literature. They are empirical, derived in this repo
# from Curt's computed term files - see their comments for provenance.
#---------------------------------------------------------------------------

MU_B = 9.2740100783e-24   #Bohr magneton, J/T
H_PL = 6.62607015e-34     #Planck constant, J s
C_L  = 2.99792458e8       #speed of light, m/s

#Fallback coefficient for lines whose terms are not LS coupled (jk / bracket
#notation), where no Lande g-factor is defined. This is max(width/lambda^2)
#measured across Curt's computed term files - set by C III 465.025 nm, whose
#1001 pm envelope is the widest in the dataset - so scaling it by lambda^2
#reproduces the conservative "widest line we have seen" width at any
#wavelength. Plain max width would badly under-estimate the near-IR neutrals,
#since Zeeman splitting in wavelength grows as lambda^2.
C_MAX_JK = 4.63e-6        #1/nm

#1% of peak on a Gaussian occurs at 1.288*FWHM either side of line centre.
#LowHighCent measures Curt's envelopes at that same 1% threshold, so adding
#this keeps estimates directly comparable with the measured widths.
_TAIL_FWHM = 1.288

_LMAP = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,'I':6,'K':7}

def Parse_Term(term):
    """
    Parse an LS term symbol into (S, L).
    Handles a leading configuration letter ('a 5D', 'x 5F*') and a trailing
    parity marker ('*'). Returns None for jk / bracket notation such as
    '2[3/2]*' or for blank entries, which have no LS g-factor.
    Inputs:
    - term: str - the term symbol, e.g. '3D', '5F*', 'a 5D', '2[3/2]*'
    Outputs:
    - (S, L): tuple of floats, or None if the term is not LS coupled
    """
    if not isinstance(term, str):
        return None
    t = term.strip().rstrip('*').strip()
    t = re.sub(r'^[a-z]\s*', '', t)          #drop 'a ', 'x ', 'd ' prefixes
    m = re.fullmatch(r'(\d+)([SPDFGHIK])', t)
    if not m:
        return None                          #jk/bracket notation or unparseable
    mult = int(m.group(1))
    return ((mult - 1)/2.0, float(_LMAP[m.group(2)]))

def Parse_J(j):
    """
    Parse a J value which may be written as a fraction ('3/2') or integer.
    Returns None if it cannot be parsed.
    """
    try:
        return float(Fraction(str(j).strip()))
    except Exception:
        return None

def Lande_g(J, S, L):
    """
    Lande g-factor for an LS-coupled level. J = 0 levels do not split.
    """
    if J == 0:
        return 0.0
    return 1.0 + (J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))

def Zeeman_Pattern_Extent(jLow, gLow, jUpp, gUpp):
    """
    Largest |m_upp*g_upp - m_low*g_low| over the allowed dipole components
    (delta m = 0, +-1), i.e. how far the outermost Zeeman component sits from
    line centre in units of the Lorentz unit.
    Inputs:
    - jLow, jUpp: float - the J values of the lower and upper levels
    - gLow, gUpp: float - the corresponding Lande g-factors
    Outputs:
    - extent: float - max shift in Lorentz units (mu_B*B/hc)
    """
    extent = 0.0
    ml = -jLow
    while ml <= jLow + 1e-9:
        for dm in (-1, 0, 1):
            mu = ml + dm
            if abs(mu) <= jUpp + 1e-9:
                extent = max(extent, abs(mu*gUpp - ml*gLow))
        ml += 1
    return extent

def Est_Zeeman_Width(wvl, termLow, jLow, termUpp, jUpp, bField = 12.5,
                     fwhm = 0.01, safety = 1.9, cMax = C_MAX_JK):
    """
    Estimate the full Zeeman envelope width (nm) of each line, matching the
    definition LowHighCent measures on Curt's spectra (outermost components
    plus the 1% instrument tails).

    Two tiers:
    - LS-coupled terms: analytic Lande pattern. Validated against the 80 lines
      Curt has computed, where measured/analytic has a median of 0.99 but
      scatters up to ~2.9 because neighbouring J components blend into one
      envelope. The safety factor (95th percentile of that ratio) keeps the
      estimate on the conservative side.
    - jk / bracket-coupled or unparseable terms: no valid g-factor exists, so
      fall back to cMax*lambda^2, the widest width/lambda^2 seen in the
      computed data (see C_MAX_JK).

    Inputs:
    - wvl: nparray (n) - line wavelengths in nm
    - termLow, termUpp: nparray (n) of str - lower/upper term symbols
    - jLow, jUpp: nparray (n) - lower/upper J values (may be '3/2' style str)
    - bField: float - field strength in tesla (12.5 T = 125 kG, as Curt used)
    - fwhm: float - instrument FWHM in nm used in Curt's calculation
    - safety: float - multiplier applied to the analytic tier only
    - cMax: float - fallback max(width/lambda^2) in 1/nm for non-LS terms
    Outputs:
    - width: nparray (n) - estimated envelope width in nm
    - method: nparray (n) of str - 'analytic-Lande' or 'lambda2-scaled max'
    """
    wvl = np.atleast_1d(np.asarray(wvl, dtype=float))
    termLow = np.atleast_1d(np.asarray(termLow, dtype=object))
    termUpp = np.atleast_1d(np.asarray(termUpp, dtype=object))
    jLow = np.atleast_1d(np.asarray(jLow, dtype=object))
    jUpp = np.atleast_1d(np.asarray(jUpp, dtype=object))

    #Lorentz unit mu_B*B/(hc) in 1/m, converted so that
    #dLambda[nm] = lambda[nm]^2 * kNm * shift
    kNm = (MU_B*bField/(H_PL*C_L))*1e-9
    tail = 2*_TAIL_FWHM*fwhm

    width = np.zeros(wvl.shape[0])
    method = np.empty(wvl.shape[0], dtype=object)

    for i in range(wvl.shape[0]):
        tl, tu = Parse_Term(termLow[i]), Parse_Term(termUpp[i])
        jl, ju = Parse_J(jLow[i]), Parse_J(jUpp[i])

        if tl is None or tu is None or jl is None or ju is None:
            #not LS coupled - use the conservative lambda^2-scaled maximum
            width[i] = cMax*wvl[i]**2
            method[i] = 'lambda2-scaled max'
        else:
            gl = Lande_g(jl, tl[0], tl[1])
            gu = Lande_g(ju, tu[0], tu[1])
            extent = Zeeman_Pattern_Extent(jl, gl, ju, gu)
            width[i] = safety*(2*extent*wvl[i]**2*kNm) + tail
            method[i] = 'analytic-Lande'

    return width, method

def Merge_Zeeman_Catalog(folder, xlsxFile, tol = 0.1, saveCSV = True,
                         verbose = True, **estKw):
    """
    Build the full Zeeman overlap catalog that Grade_Zeeman works from, by
    combining:
      1. Curt's MEASURED envelopes  - every peak in <folder>/wavelengths.csv,
         produced by Load_H5_Zeeman/LowHighCent from the h5 term files.
      2. ESTIMATED envelopes        - for candidate lines in the spreadsheet
         that have no computed counterpart in the h5 files at all.

    Measured always wins: a spreadsheet line is only given an estimated
    envelope if no h5 file lists it among its NIST lines. This stops
    uncomputed lines from looking like empty spectrum to Grade_Zeeman.

    Note the two inputs cover different things and that is intentional. The h5
    catalog contains every J-resolved peak Curt computed, including many not in
    the spreadsheet; the estimates cover spreadsheet lines Curt has not reached
    yet. Both are contamination sources, so both belong in the catalog
    regardless of the 1 nm downselect applied to the selectable line list.

    Inputs:
    - folder: str - folder holding the h5 term files AND wavelengths.csv
    - xlsxFile: str - the candidate line list spreadsheet
    - tol: float - nm tolerance for matching a spreadsheet line to a NIST line
    - saveCSV: bool - write zeeman_catalog.csv into folder for inspection
    - verbose: bool - print a coverage summary
    - estKw: passed through to Est_Zeeman_Width (bField, safety, cMax, ...)
    Outputs:
    - center, low, high: nparray - envelope centre and edges (nm) for every
      entry, ready for Process_Zeeman
    - isEst: nparray of bool - True where the entry is an estimate
    """

    #---- 1. measured envelopes from the h5 term files --------------------
    meas = pd.read_csv(os.path.join(folder, 'wavelengths.csv'))

    #---- which lines has Curt actually computed? -------------------------
    #Use the nist_lines groups rather than the peak centres: a term file lists
    #every J-resolved line it covers, which is what "computed" really means.
    computed = []
    for f in os.listdir(folder):
        if not f.endswith('.h5'):
            continue
        with h5py.File(os.path.join(folder, f), 'r') as h:
            el = h.attrs['element']
            ch = int(h.attrs['charge'])
            if 'nist_lines' not in h:
                continue
            obs = h['nist_lines']['observed_wavelength_nm'][:]
            ritz = h['nist_lines']['ritz_wavelength_nm'][:]
        #prefer observed, fall back to ritz (Kr II / Mo I have ritz all NaN)
        for o, r in zip(obs, ritz):
            w = o if not np.isnan(o) else r
            if not np.isnan(w):
                computed.append((el, ch, float(w)))

    #---- 2. estimated envelopes for spreadsheet lines with no h5 data ----
    xWvl, xSpec, _, xIon, _, xWidth, xMethod = Load_xlsx(xlsxFile)
    if estKw:
        #recompute with caller's settings (Load_xlsx uses the defaults)
        data = pd.read_excel(xlsxFile, sheet_name='Sheet1', header=0,
                             usecols=[1,2,3,5,6,9,10,12,13,14,16])
        xWidth, xMethod = Est_Zeeman_Width(xWvl, data['term_low'], data['j_low'],
                                           data['term_upp'], data['j_upp'], **estKw)

    eCen, eLow, eHigh, eLbl = [], [], [], []
    seen = set()
    for i in range(xWvl.shape[0]):
        el, ch, w = xSpec[i], int(xIon[i]), float(xWvl[i])
        #the spreadsheet has a few duplicated rows - do not double count them
        if (el, ch, round(w, 3)) in seen:
            continue
        seen.add((el, ch, round(w, 3)))

        if any(cel == el and cch == ch and abs(cw - w) <= tol
               for cel, cch, cw in computed):
            continue                       #measured data exists, skip estimate

        eCen.append(w)
        eLow.append(w - xWidth[i]/2.0)
        eHigh.append(w + xWidth[i]/2.0)
        eLbl.append(f'{el}_{ch}_{w:.3f}_EST_{xMethod[i]}')

    #---- 3. combine ------------------------------------------------------
    center = np.concatenate([meas['center'].values, np.asarray(eCen)])
    low    = np.concatenate([meas['low'].values,    np.asarray(eLow)])
    high   = np.concatenate([meas['high'].values,   np.asarray(eHigh)])
    isEst  = np.concatenate([np.zeros(len(meas), bool), np.ones(len(eCen), bool)])

    if verbose:
        print(f'Zeeman catalog: {len(meas)} measured peaks + {len(eCen)} estimated '
              f'envelopes = {len(center)} entries')
        if len(eCen):
            print(f'  estimated widths (nm): min {np.min(np.asarray(eHigh)-np.asarray(eLow)):.3f} '
                  f'median {np.median(np.asarray(eHigh)-np.asarray(eLow)):.3f} '
                  f'max {np.max(np.asarray(eHigh)-np.asarray(eLow)):.3f}')

    if saveCSV:
        pd.DataFrame({'file': list(meas['file']) + eLbl,
                      'low': low, 'high': high, 'center': center,
                      'estimated': isEst}).to_csv(
            os.path.join(folder, 'zeeman_catalog.csv'), index=False)

    return center, low, high, isEst

def Load_xlsx(filename, prevTokCol = 'Prev_Tok'):
    data = pd.read_excel(filename, sheet_name='Sheet1',header = 0,
                         usecols = [1,2,3,5,6,9,10,12,13,14,15,16])
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
    
    #Estimated Zeeman envelope width for every line, from the atomic terms.
    #This replaces the old random-filler widths, which were unseeded (so runs
    #were not reproducible) and assigned widths sampled from unrelated lines.
    #The 'Zee_width (nm)' column in the spreadsheet is deliberately NOT used:
    #it only covers 22 N II/N III lines and disagrees with the current h5
    #calculation by 0.1x-3.1x, i.e. it predates the term-based dataset.
    #Measured h5 widths win over these estimates in Merge_Zeeman_Catalog.
    width, method = Est_Zeeman_Width(wvl, data['term_low'], data['j_low'],
                                     data['term_upp'], data['j_upp'])

    #Previously fielded on a tokamak. Marked 'Y' in the spreadsheet, blank
    #otherwise, so anything that is not a 'Y' counts as unused. Returned as a
    #real boolean array - this used to be hardcoded to all-False, which made
    #Grade_PrevTok score 0 for every stack.
    if prevTokCol in data.columns:
        prevTok = (data[prevTokCol].astype(str).str.strip().str.upper()
                   == 'Y').to_numpy()
    else:
        print(f"WARNING: no '{prevTokCol}' column in {filename}; "
              f"previous-tokamak score will be 0 for every stack")
        prevTok = np.zeros(wvl.shape[0], dtype=bool)

    return wvl, spec, species, ion, prevTok,width,method


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

    Note: estimated Zeeman widths are deliberately NOT returned here. The
    overlap catalog needs every line in the spreadsheet, including ones this
    function drops in the 1 nm downselect (a dropped line still emits and can
    still contaminate a neighbour, it just is not separately selectable), so
    Merge_Zeeman_Catalog goes to Load_xlsx for the full list instead.
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

    wvl, spec, species, ion, prevTok,_,_ = Load_xlsx(filename)



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



    #Load_xlsx already returns a boolean. The old code here compared this
    #array against the string 'Y', which is False for a bool array and so
    #zeroed the flag even once the column was being read. Kept tolerant of a
    #string column in case the loader is ever pointed at the older sheets.
    if prevTok.dtype == bool:
        prevOut = prevTok
    else:
        prevOut = (np.char.upper(prevTok.astype(str).astype('U')) == 'Y')

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



def Filter_Stacks(stackL,wvl,spec,ion,prevTok, force = None, tol = 0.05):
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
    - force: dict or None - pin a species to one or more specific lines, e.g.
      {'He': 471.315} or {'He': [471.315, 587.562]}. This is applied BEFORE the
      combinations are generated, so it divides the total rather than filtering
      afterwards: pinning He drops its 10 candidates to 1 and the whole product
      shrinks 10x. Use this, not Grade_Stack's force, whenever you can - it is
      the difference between a search that fits in memory and one that does not.
    - tol: float - nm tolerance when matching a forced wavelength to the list
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

    #a forced species that is not in any stack would otherwise be ignored in
    #silence, so a typo would quietly grade the unconstrained search
    if force:
        stray = [s for s in force if s not in set(uspec)]
        if stray:
            raise ValueError(f"force names {stray} which are not in stackL "
                             f"(stack species: {', '.join(uspec)})")

    #candidate line indices per species, after any forcing
    keep = []
    for s in uspec:
        idxS = np.flatnonzero(spec == s)
        if len(idxS) == 0:
            raise ValueError(f"species {s!r} appears in stackL but has no lines "
                             f"in the line list; available: "
                             f"{', '.join(sorted(set(spec)))}")
        if force and s in force:
            want = np.atleast_1d(np.asarray(force[s], dtype=float))
            sel = []
            for w in want:
                j = int(np.argmin(np.abs(wvl[idxS] - w)))
                if abs(wvl[idxS][j] - w) > tol:
                    raise ValueError(f"forced line {w:.3f} nm for {s} is not in "
                                     f"the list (nearest {wvl[idxS][j]:.3f} nm, "
                                     f"tol {tol} nm)")
                sel.append(idxS[j])
            idxS = np.array(sorted(set(sel)))
            print(f'  forcing {s} to {", ".join(f"{wvl[k]:.3f}" for k in idxS)} nm')
        keep.append(idxS)

    num = np.array([len(k) for k in keep], dtype=int)

    #multiply the number of lines for each species

    tot = int(np.prod(num.astype(object)))

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
        
        #keep[i] already has any forced restriction applied
        kk = keep[i]
        wvlS = wvl[kk]
        ionS = s+' '+ion[kk]
        prevTokS = prevTok[kk]



        #repeat the wvlS for the number of repeats
        outWvl[i,:] = Pattern(wvlS,repeats, tiles)
        outAS[i,:] = Pattern(ionS,repeats, tiles)
        outPT[i,:] = Pattern(prevTokS,repeats, tiles)


    print(f'Combinations generated in {time()-t:.2f} seconds')

    return outWvl, outAS, outPT

def Line_Envelopes(lineWvl, cen, low, high, tol = 0.5):
    """
    Find each candidate line's OWN Zeeman envelope in the catalog.

    Prefers an entry that actually contains the line (nearest centre among
    those), otherwise falls back to the nearest centre within tol. Lines with
    no match get NaN, which Zeeman_Line_Scores treats as "no Zeeman data".

    Inputs:
    - lineWvl: nparray (n) - candidate line wavelengths in nm
    - cen, low, high: nparray (m) - catalog envelope centres and edges
    - tol: float - max nm from a catalog centre to still call it the same line
    Outputs:
    - lineLow, lineHigh: nparray (n) - the line's own envelope, NaN if none
    """
    lineWvl = np.asarray(lineWvl, dtype=float)
    cen, low, high = np.asarray(cen), np.asarray(low), np.asarray(high)

    lineLow = np.full(lineWvl.shape[0], np.nan)
    lineHigh = np.full(lineWvl.shape[0], np.nan)
    for i, w in enumerate(lineWvl):
        inside = (low <= w) & (high >= w)
        if inside.any():
            j = np.flatnonzero(inside)[np.argmin(np.abs(cen[inside] - w))]
        else:
            j = int(np.argmin(np.abs(cen - w)))
            if abs(cen[j] - w) > tol:
                continue                    #no data for this line
        lineLow[i], lineHigh[i] = low[j], high[j]
    return lineLow, lineHigh

def Zeeman_Line_Scores(lineWvl, cen, low, high, mode = 'envelope', tol = 0.5):
    """
    Exact Zeeman blend score for each candidate line. The score is -(count-1),
    where count includes the line's own envelope: 0 when the line sits alone,
    -1 when one other line blends with it, and so on. A line with no Zeeman
    data at all scores 0, so missing data is neutral rather than rewarded.

    Two definitions of "blended", selected by `mode`:

    - 'envelope' (default, CONSERVATIVE): the line's own Zeeman envelope
      INTERSECTS another line's envelope. This is the right question for a
      filter stack - if two broadened profiles touch at all, a bandpass around
      one admits some of the other, even when neither line centre is buried.
      Catches cases like Fe III 411.986, whose envelope overlaps B II 412.193
      while its centre sits just outside B II's envelope.

    - 'center': another line's envelope contains this line's CENTRE, i.e. the
      line is substantially buried rather than merely touching. Less strict,
      so more lines come back clean. Kept as a fallback for when the
      conservative test leaves too few acceptable stacks to choose between.

    On the current data 'envelope' flags 40 of 106 selectable lines and
    'center' flags 25, so the conservative mode still discriminates rather
    than condemning everything.

    IMPORTANT: cen/low/high must be the FULL catalog from Merge_Zeeman_Catalog
    - every measured h5 peak plus an estimated envelope for every uncomputed
    spreadsheet line. Contamination has to be counted against every line that
    physically emits, not just the subset that survives Load_data's 1 nm
    downselect: a line dropped there is merely not separately SELECTABLE, it
    still sits in the plasma and still blends into its neighbours.

    Inputs:
    - lineWvl: nparray (n) - candidate line wavelengths in nm (exact, float64)
    - cen, low, high: nparray (m) - catalog envelope centres and edges in nm
    - mode: 'envelope' (conservative) or 'center'
    - tol: float - passed to Line_Envelopes when mode='envelope'
    Outputs:
    - score: nparray (n) float32 - per-line blend score, <= 0
    - count: nparray (n) int - envelopes blended with each line, own included
    """
    lineWvl = np.asarray(lineWvl, dtype=float)
    low, high = np.asarray(low), np.asarray(high)

    if mode == 'center':
        count = ((lineWvl[:,None] >= low[None,:]) &
                 (lineWvl[:,None] <= high[None,:])).sum(1)
    elif mode == 'envelope':
        lLow, lHigh = Line_Envelopes(lineWvl, cen, low, high, tol = tol)
        #two intervals intersect unless one ends before the other starts.
        #NaN (no data for this line) compares False, giving count 0 -> score 0
        count = ((lLow[:,None] <= high[None,:]) &
                 (lHigh[:,None] >= low[None,:])).sum(1)
    else:
        raise ValueError(f"mode must be 'envelope' or 'center', got {mode!r}")

    score = np.minimum(1 - count, 0).astype(np.float32)
    return score, count

def Zeeman_Line_LUT(lineWvl, lineSpec, uspec, cen, low, high,
                    mode = 'envelope', tol = 0.5):
    """
    Build a float16-indexed lookup table of per-line Zeeman scores, one row per
    species, so Grade_Zeeman can score a stack with a plain gather.

    Filter_Stacks stores the combination wavelengths as float16, so every value
    in row i of that array is the float16 image of one of species uspec[i]'s
    lines. Reinterpreting those 16 bits as a uint16 gives a direct index into a
    65536-entry table - exact, and far cheaper than searching a wavelength grid.

    The table is built PER SPECIES rather than globally because two lines of
    different species can round to the same float16 value (e.g. He I 667.815
    and Ne I 667.828). Within one species Load_data's downselect guarantees
    lines are more than 1 nm apart, comfortably wider than float16 spacing
    (<= 0.5 nm here), so a per-species table can never be ambiguous.

    Inputs:
    - lineWvl: nparray (n) - candidate line wavelengths (exact)
    - lineSpec: nparray (n) of str - the element of each candidate line
    - uspec: sequence (k) of str - species in row order of the combination array
    - cen, low, high: nparray (m) - full catalog (see Zeeman_Line_Scores)
    - mode: 'envelope' (conservative, default) or 'center' - see Zeeman_Line_Scores
    - tol: float - passed through to Zeeman_Line_Scores
    Outputs:
    - lut: nparray (k, 65536) float32 - lut[i, code] = score of that line
    - score: nparray (n) float32 - the per-line scores, for inspection
    - count: nparray (n) int - envelopes blended with each line, for inspection
    """
    score, count = Zeeman_Line_Scores(lineWvl, cen, low, high, mode = mode, tol = tol)
    lineWvl = np.asarray(lineWvl, dtype=float)
    lineSpec = np.asarray(lineSpec)

    lut = np.zeros((len(uspec), 65536), dtype=np.float32)
    for i, s in enumerate(uspec):
        m = lineSpec == s
        codes = np.ascontiguousarray(lineWvl[m].astype(np.float16)).view(np.uint16)
        if len(np.unique(codes)) != len(codes):
            print(f'WARNING: float16 collision within species {s}; '
                  f'Zeeman scores for it may be wrong')
        lut[i, codes] = score[m]
    return lut, score, count

def Grade_Zeeman(wvl, lut, chunkN = 20_000_000):
    """
    Score each stack on how much its lines are blended with other survey lines
    by Zeeman broadening. 0 is perfect (every chosen line sits alone in its own
    envelope) and each additional line blended with a chosen line costs -1, so
    the score is <= 0.

    This is an exact evaluation: the per-line scores in `lut` were computed by
    counting catalog envelopes containing each line's true wavelength (see
    Zeeman_Line_Scores), so scoring a stack is just a gather and a sum.

    It replaces an earlier version that searched a uniform 0.25 nm wavelength
    grid on the GPU for the nearest grid point. That was both slow (cost grew
    linearly with grid size, so a fine grid was unaffordable) and wrong at this
    resolution: overlap bands are only ~0.2-0.4 nm wide, so the nearest grid
    point routinely fell outside the very overlap it was meant to detect. Ni I
    471.442 sits inside He I's 471.180-471.477 envelope, but the nearest grid
    point was 471.513 - past its edge - so the whole W/Mo/Fe/Ni/Cu/Al stack
    scored a flat 0. Grade_Zeeman_Grid below is kept for reference.

    Inputs:
    - wvl: nparray (k, nCombos) float16 - stack wavelengths from Filter_Stacks
    - lut: nparray (k, 65536) float32 - per-species score table, rows aligned
      with wvl, from Zeeman_Line_LUT
    - chunkN: int - combinations per chunk, to bound peak memory
    Outputs:
    - out: nparray (nCombos) float32 - Zeeman score per stack, <= 0
    """
    if wvl.dtype != np.float16:
        raise TypeError('Grade_Zeeman expects the float16 combination array '
                        'that Filter_Stacks produces, so the lut indices line up')

    ncomb = wvl.shape[-1]
    out = np.zeros(ncomb, dtype=np.float32)

    for s in range(0, ncomb, chunkN):
        e = min(s + chunkN, ncomb)
        #reinterpret the float16 bits as uint16 to index the table directly
        codes = np.ascontiguousarray(wvl[:, s:e]).view(np.uint16)
        for r in range(wvl.shape[0]):
            out[s:e] += lut[r][codes[r]]

    return out

def Grade_Zeeman_Grid(wvl, overWvl, Overlap, plot=False, chunkN = 20_000_000):
    """
    LEGACY nearest-grid-point version of Grade_Zeeman, kept for reference and
    for the Prep_Zeeman path in Grade_Stack. Prefer Grade_Zeeman: this one is
    only as accurate as the wavelength grid it is handed, and at the 0.25 nm
    spacing Grade_Stack used it silently missed most real overlaps.

    Memory-efficient: processes the combination axis in chunks of chunkN so no
    single host-pinned/GPU allocation exceeds a few hundred MB (a 6-species x
    98M combination stack needs a 4 GB pinned transfer in one shot, which the
    driver refuses).

    Per-line contribution is clipped at 0. Overlap is -n where n envelopes
    cover a wavelength, so a line alone in its own envelope reads -1 and
    contributes -1+1 = 0. A line at a wavelength with NO catalog envelope at
    all reads 0, which without the clip would contribute +1 - i.e. the old
    scoring actively REWARDED lines whose Zeeman width had never been
    calculated, and the metal stacks scored well largely because Fe/Ni/Cu/Al
    had no data. Clipping makes "no data" merely neutral rather than better
    than a clean, well-characterised line.
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
        #+1 cancels each line's own envelope; clipping at 0 stops lines with no
        #Zeeman data (overlap 0) from scoring better than isolated lines
        out[s:e] = np.minimum(np.take(Overlap, idxArr) + 1, 0).sum(0)

        del idx, smllD, wvlT, diff, mask
        dr.flush_malloc_cache()  #release cached GPU/pinned memory before the next chunk
        print(f'chunk {e}/{ncomb} done')

    return out

def Grade_Wvl(wvl, clip = 30, gBS = 2.0, dichFloor = 0.85, bsScore = 0.25):
    """
    Grade a stack on how easily its lines can be optically separated onto
    separate channels. Every adjacent pair (in sorted wavelength order) needs
    its own split, so we score each gap and average over the (nLines-1) splits.

    Separation is never impossible, so this is a soft cost gradient with a
    floor, not a hard pass/fail:
    - gap >= clip (30 nm): standard/cheap dichroic, full light -> score 1.0
    - gBS <= gap < clip:   dichroic still works at full light but needs a
      sharper/pricier edge as the gap shrinks -> linear 1.0 down to dichFloor
      (a "this costs more money" penalty, light throughput still ~full)
    - gap < gBS:           too tight for any practical dichroic edge, fall back
      to a 50/50 beamsplitter + bandpass -> flat bsScore floor (acceptable, but
      loses light). The step down at gBS reflects the hardware regime change.

    Score per stack is in [bsScore, 1]. Averaging (not min) is used on purpose:
    one beamsplitter pair is acceptable, so a single tight gap should only dent
    the score, while a stack full of tight gaps (lots of lost light) should
    score much lower.

    Inputs:
    - wvl: nparray (nLines, nCombos) - the wavelengths of the lines in each stack
    - clip: float - gap (nm) at/above which a cheap dichroic suffices (score 1)
    - gBS: float - gap (nm) below which no practical dichroic edge exists and a
      beamsplitter is required (dichroic->beamsplitter crossover)
    - dichFloor: float - score at the gBS end of the dichroic regime (encodes
      the added cost of the sharpest practical dichroic; ~full light so <1 but high)
    - bsScore: float - score floor for the beamsplitter fallback (~light kept in
      a 50/50 split, hence ~0.5)
    Outputs:
    - out: nparray (nCombos) - mean separation score per stack, in [bsScore, 1]
    """

    #float32: ~2x faster than float16 here (no vectorized f16 arithmetic path)
    #and restores real sub-nm resolution lost to float16's 0.25-0.5 nm quantization
    sWvl = np.sort(wvl.astype(np.float32), axis = 0)
    gaps = np.diff(sWvl, axis = 0)  #adjacent gaps (nm) in sorted order

    #dichroic regime: linear from 1.0 at clip down to dichFloor at gBS
    frac = np.clip((gaps - gBS) / (clip - gBS), 0.0, 1.0)
    dich = dichFloor + (1.0 - dichFloor) * frac

    #below gBS the split needs a beamsplitter -> flat floor
    score = np.where(gaps >= gBS, dich, np.float32(bsScore))

    out = score.mean(0)  #mean over the (nLines-1) splits, comparable across stack sizes

    return out

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

def Grade_Stack(stackL, wvl, atomicS, prevTok,filename,lowMem = True,
                zFolder = None, xlsxFile = None, zMode = 'envelope',
                force = None, **mergeKw):
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
    - filename: str - wavelengths.csv, used only on the legacy path (see zFolder)
    - zFolder: str or None - folder of h5 term files. If given together with
      xlsxFile, the Zeeman catalog is built by Merge_Zeeman_Catalog (measured
      envelopes plus estimates for uncomputed lines). If None, falls back to
      the legacy measured-only catalog read from `filename`.
    - xlsxFile: str or None - candidate line list spreadsheet, needed with zFolder
    - zMode: 'envelope' (default, conservative: Zeeman envelopes intersect) or
      'center' (line centre buried under another envelope). Switch to 'center'
      if the conservative test leaves too few clean stacks to choose between.
    - force: dict or None - pin a species to one or more lines, e.g.
      {'He': 471.315}. Combinations that do not use a pinned line get -inf so
      they sort last, leaving the ranking to the allowed ones. This filters
      AFTER the combinations exist, so it costs no less memory - prefer
      Filter_Stacks(force=...), which shrinks the product instead. Use this
      only to re-slice an already generated (or loaded) set.
    - mergeKw: passed to Merge_Zeeman_Catalog (tol, bField, safety, cMax, ...)
    Outputs:
    - out: nparray - (wavelength combinations, 3) - the scores for each stack, where the first column is the wavelength score,
     the second column is the previous tokamak usage score and the third column is the UV score
    """

    #calcualte three metrics
    #1) median wavelength range
    #2) previous tokamak usage
    #3) number of lines below a certain wavelength


    #spec = np.array([s.split(' ')[0] for s in atomicS])

    out1 = np.zeros((wvl.shape[-1],4),dtype = np.float16)


    #all the same for each row, so we can just use the first row
    cspec = np.array([s.split(' ')[0] for s in atomicS[:,0]])

    #setup the Zeeman scoring
    if zFolder is not None and xlsxFile is not None:
        #measured h5 envelopes + estimated envelopes for lines Curt has not
        #computed, so uncomputed lines are not invisible to Grade_Zeeman.
        #The catalog spans the FULL spreadsheet, not the downselected lines.
        cen, low, high, _ = Merge_Zeeman_Catalog(zFolder, xlsxFile, **mergeKw)

        #exact per-line scores, looked up by float16 bit pattern at grade time
        lWvl, lSpec, _, _ = Load_data(xlsxFile)
        lut, lScore, lCount = Zeeman_Line_LUT(lWvl, lSpec, cspec, cen, low, high,
                                              mode = zMode)
        print(f"Zeeman ({zMode} mode): {int((lScore!=0).sum())} of {len(lWvl)} "
              f"selectable lines are blended with another line "
              f"(worst {int(-lScore.min())} overlapping)")
        if (lScore == 0).sum() == 0:
            print("  WARNING: every selectable line is blended. Consider "
                  "zMode='center' for the less strict centre-buried test.")
        useLUT = True
    else:
        #legacy path: measured h5 envelopes only, nearest point on a 0.25 nm grid
        wvlOvr,ovrlp = Prep_Zeeman(filename, wvlD = 0.25, plot = True)
        useLUT = False

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


        out1[:,0] += Grade_Wvl(cwvl)
        print('wvl done')
        out1[:,1] += Grade_PrevTok(cpt)
        print('Prev Tok done')
        out1[:,2] += Grade_UV(cwvl)
        print('UV done')
        if useLUT:
            out1[:,3] += Grade_Zeeman(cwvl, lut[idx[i]])
        else:
            out1[:,3] += Grade_Zeeman_Grid(cwvl,wvlOvr,ovrlp)
        print('Zeeman done')

        print(f'Stack {i+1} graded in {time()-t:.2f} seconds')

    #post-hoc pinning: sink every combination that does not use a forced line
    if force:
        ok = np.ones(wvl.shape[-1], dtype=bool)
        for s, w in force.items():
            rows = np.flatnonzero(cspec == s)
            if len(rows) == 0:
                raise ValueError(f"forced species {s!r} is not in this stack set")
            #the combination array is float16, so compare against the float16
            #image of the requested wavelengths
            want = np.unique(np.atleast_1d(np.asarray(w)).astype(np.float16))
            ok &= np.isin(wvl[rows[0]], want)
        print(f'force: {int(ok.sum()):,} of {len(ok):,} combinations keep the '
              f'pinned lines; the rest are set to -inf')
        if not ok.any():
            raise ValueError('force removed every combination - check the '
                             'wavelengths match the candidate list exactly')
        out1[~ok] = -np.inf

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

def Load_H5_Spectra(folder):
    """
    Load every term file's normalised Zeeman spectrum once, for plotting.
    Unlike Load_H5_Zeeman this keeps the individual spectra rather than
    summing them, so each can be drawn in its own colour.
    Inputs:
    - folder: str - folder of h5 term files
    Outputs:
    - spectra: list of dicts with w, s (normalised 0-1), cen (peak centres),
      el, ch and a display label
    """
    spectra = []
    for f in sorted(x for x in os.listdir(folder) if x.endswith('.h5')):
        with h5py.File(os.path.join(folder, f), 'r') as h:
            w = h['wave_air'][:]
            s = h['signal'][:]
            el = h.attrs['element']
            ch = int(h.attrs['charge'])
            src = float(h.attrs['source_line_nm'])
        si = np.argsort(w)
        w, s = w[si], s[si]
        if np.any(np.isnan(s)) or s.max() == s.min():
            continue
        s = (s - s.min())/(s.max() - s.min())
        low, high, cen = LowHighCent(w, s)
        spectra.append(dict(w=w, s=s, cen=cen, el=el, ch=ch,
                            lbl=f'{el} {ch} {src:.1f}'))
    return spectra

def Plot_Zeeman_Spectra(ax, spectra, selWvl, window = 3.0, height = 3.0,
                        cmap = 'hsv', nCycle = 9, label = True):
    """
    Draw the Zeeman spectra sitting near a stack's selected lines, each in its
    own colour from a cycling colour wheel.

    Only the parts of a spectrum within `window` nm of a selected line are
    drawn - plotting the whole 300-900 nm survey in every panel buries the
    detail that matters, which is what sits right next to the chosen lines.
    Everything outside the window is set to NaN so matplotlib breaks the trace
    rather than joining across the gap.

    Inputs:
    - ax: matplotlib axis to draw on
    - spectra: list from Load_H5_Spectra
    - selWvl: nparray - the wavelengths selected for this stack
    - window: float - nm either side of a selected line to draw
    - height: float - plot height for a fully normalised (1.0) spectrum
    - cmap: str - cyclic colour map used to tell neighbouring lines apart
    - nCycle: int - how many colours before the wheel repeats
    - label: bool - annotate each drawn spectrum with its term-file label
    Outputs:
    - drawn: list of the spectra that were close enough to plot
    """
    selWvl = np.atleast_1d(np.asarray(selWvl, dtype=float))
    cm = plt.get_cmap(cmap)

    #a spectrum is worth drawing if any of its peaks is near a selected line
    drawn = [sp for sp in spectra
             if len(sp['cen']) and
             np.any(np.abs(np.asarray(sp['cen'])[:,None] - selWvl[None,:]) <= window)]

    def nearestPeak(sp):
        """The spectrum's peak lying closest to any of the selected lines."""
        cen = np.asarray(sp['cen'])
        return float(cen[np.abs(cen[:,None] - selWvl[None,:]).min(1).argmin()])

    #order left to right so the staggered labels read in wavelength order
    drawn.sort(key=nearestPeak)

    for k, sp in enumerate(drawn):
        #hsv straight off the wheel puts near-white yellows on a white page,
        #so pull the value down a little to keep every colour legible
        r, g, b, _ = cm((k % nCycle)/nCycle)
        col = (r*0.82, g*0.82, b*0.82)

        near = np.any(np.abs(sp['w'][:,None] - selWvl[None,:]) <= window, axis=1)
        y = np.where(near, sp['s']*height, np.nan)
        ax.plot(sp['w'], y, color=col, lw=1.1, zorder=6)
        ax.fill_between(sp['w'], 0, np.nan_to_num(y, nan=0.0),
                        where=near, color=col, alpha=0.28, lw=0, zorder=5)
        if label:
            #label at the peak nearest a selected line, staggered over three
            #rows so tightly spaced neighbours stay readable
            ax.text(nearestPeak(sp), height*(1.04 + 0.30*(k % 3)), sp['lbl'],
                    color=col, fontsize=6.5, ha='center', va='bottom', zorder=7)
    return drawn

def Plot_Zeeman_Estimates(ax, estCen, estLow, estHigh, estLbl, selWvl,
                          window = 3.0, height = 3.0, cmap = 'hsv',
                          nCycle = 9, cStart = 0, label = True):
    """
    Draw the ESTIMATED Zeeman envelopes near a stack's selected lines.

    These are the candidate lines Curt has not computed - no h5 file, so no
    measured profile to plot. Without them a panel looks empty next to, say,
    Ni I or Cu I, which reads as "nothing nearby" when the truth is "nothing
    calculated". They are drawn as flat hatched boxes spanning the estimated
    envelope rather than as spectra, precisely because the line SHAPE is not
    known - only its width, from Est_Zeeman_Width.

    Inputs:
    - ax: matplotlib axis to draw on
    - estCen, estLow, estHigh: nparray - estimated envelope centre and edges
    - estLbl: sequence of str - display label per estimated entry
    - selWvl: nparray - the wavelengths selected for this stack
    - window: float - nm either side of a selected line to draw
    - height: float - the height a full-scale measured spectrum would reach
    - cmap, nCycle: cycling colour wheel, matched to Plot_Zeeman_Spectra
    - cStart: int - colour index to start from, so estimates continue the
      cycle rather than repeating the measured spectra's colours
    - label: bool - annotate each box
    Outputs:
    - drawn: list of indices actually plotted
    """
    selWvl = np.atleast_1d(np.asarray(selWvl, dtype=float))
    cm = plt.get_cmap(cmap)
    estCen = np.asarray(estCen); estLow = np.asarray(estLow); estHigh = np.asarray(estHigh)

    near = np.flatnonzero(np.min(np.abs(estCen[:,None] - selWvl[None,:]), axis=1) <= window)
    near = near[np.argsort(estCen[near])]

    for k, ii in enumerate(near):
        r, g, b, _ = cm(((cStart + k) % nCycle)/nCycle)
        col = (r*0.82, g*0.82, b*0.82)
        w = estHigh[ii] - estLow[ii]
        #flat box: the width is estimated, the profile is unknown
        ax.add_patch(plt.Rectangle((estLow[ii], 0), max(w, 0.02), height*0.5,
                                   facecolor=col, alpha=0.22, edgecolor=col,
                                   ls='--', lw=1.0, hatch='///', zorder=4))
        ax.plot([estCen[ii], estCen[ii]], [0, height*0.5], color=col,
                lw=1.0, ls='--', zorder=5)
        if label:
            ax.text(estCen[ii], height*(0.54 + 0.30*(k % 3)), estLbl[ii],
                    color=col, fontsize=6.5, ha='center', va='bottom',
                    style='italic', zorder=7)
    return list(near)

def Plot_Stack(scores,stackL,wvl,atomicS,prevTok,wvlW = 1, uvSW = 1, prevTokW = 1,zW = 1,
               zFilename = 'sparc_line_ids_widths_v0.xlsx',
               zFolder = 'split_ext_field_50_terms_h5',
               xlsxFile = 'sparc_line_ids_widths_v0.xlsx', window = 3.0,
               specH = 3.0, cmap = 'hsv', nTop = None, showEst = True,
               printSel = True):
    """
    Plot the scores for each stack, best first.

    Each panel shows the stack's selected lines (Plot_Lines) over the actual
    Zeeman spectra of whatever lies within `window` nm of them, one colour per
    neighbouring term set. This replaces the single black summed-overlap trace
    that used to run along the bottom: that curve spanned the whole survey and
    could not show WHICH line was crowding a selection.

    Inputs:
    - scores: nparray- the scores for each stack
    - stackL: list of lists- the stacks to plot
    - wvl: nparray- the wavelengths for each stack
    - atomicS: nparray- the atomic states for each stack
    - prevTok: nparray- the previous tokamak usage for each stack
    - zFolder: str - folder of h5 term files supplying the spectra
    - xlsxFile: str - candidate line list, used to build the estimated
      envelopes for lines with no h5 file
    - window: float - nm either side of a selected line to draw spectra for
    - specH: float - plot height of a fully normalised spectrum
    - cmap: str - cycling colour wheel used to separate neighbouring spectra
    - nTop: int or None - only plot this many best stacks (None = all, which
      for a 98M combination run is effectively endless)
    - showEst: bool - also draw estimated envelopes for lines outside the h5
      set, so an empty neighbourhood is not mistaken for a clean one
    - printSel: bool - print the wavelength chosen for each species in each
      stack, sorted blue to red (the order a dichroic chain would split them)
    """
    #individual spectra, kept separate so each can carry its own colour
    spectra = Load_H5_Spectra(zFolder)

    #Exact candidate wavelengths. The combination array is float16, whose
    #resolution is 0.25-0.5 nm here, so printing straight from it would report
    #e.g. He 471.25 for a line that is really at 471.315. Matching each
    #selection back to the candidate list recovers the true value.
    lWvl = lSpec = lIon = lPrev = None
    if printSel:
        try:
            lWvl, lSpec, lIon, lPrev = Load_data(xlsxFile)
        except Exception as e:
            print(f'Plot_Stack: could not read {xlsxFile} for exact '
                  f'wavelengths ({type(e).__name__}); printing float16 values')

    def Exact_Line(w16, sp):
        """Recover the true wavelength/ion/prev-tok behind a float16 selection."""
        if lWvl is None:
            return float(w16), '', False
        m = (lSpec == sp)
        if not m.any():
            return float(w16), '', False
        k = int(np.argmin(np.abs(lWvl[m] - float(w16))))
        return float(lWvl[m][k]), str(lIon[m][k]), bool(lPrev[m][k])

    #estimated envelopes for the candidate lines with no h5 spectrum
    estCen = estLow = estHigh = np.array([])
    estLbl = []
    if showEst:
        Merge_Zeeman_Catalog(zFolder, xlsxFile, verbose=False)
        cat = pd.read_csv(os.path.join(zFolder, 'zeeman_catalog.csv'))
        est = cat[cat['estimated']]
        estCen = est['center'].to_numpy()
        estLow = est['low'].to_numpy()
        estHigh = est['high'].to_numpy()
        #'Ni_1_471.442_EST_analytic-Lande' -> 'Ni 1 471.4 est'
        for f_ in est['file']:
            p = str(f_).split('_')
            try:
                estLbl.append(f'{p[0]} {p[1]} {float(p[2]):.1f} est')
            except (IndexError, ValueError):
                estLbl.append(str(f_))
        print(f'Plot_Stack: {len(spectra)} measured spectra + '
              f'{len(estCen)} estimated envelopes available')

    scores[:,0] *= wvlW
    scores[:,1] *= prevTokW
    scores[:,2] *= uvSW
    scores[:,3] *= zW
    weightS = np.sum(scores, axis=1)

    #sort the list and order from highest to lowest
    sorted_indices = np.argsort(weightS)[::-1]
    if nTop is not None:
        sorted_indices = sorted_indices[:nTop]

    cspec = np.array([s.split(' ')[0] for s in atomicS[:,0]])

    #find the indexes of the speces in the stackL
    idx = [np.searchsorted(cspec, substack) for substack in stackL]


    for rank, i in enumerate(sorted_indices, 1):
        #squeeze=False keeps this a 2-D array even for a single stack, so the
        #a[j] / a[0] / iteration below work; plt.subplots(1,1) would otherwise
        #hand back a bare Axes that is not subscriptable
        f,a = plt.subplots(len(stackL),1,sharex = False,squeeze = False)
        a = a[:,0]

        subWvl = wvl[:,i]
        subSpec = atomicS[:,i]

        if printSel:
            print(f'\n=== rank {rank}  (combination {i})  total {weightS[i]:.2f}'
                  f'  |  wvl {scores[i,0]:.2f}  uv {scores[i,2]:.2f}'
                  f'  prevTok {scores[i,1]:.2f}  Z {scores[i,3]:.2f} ===')

        for j in range(len(stackL)):
            cwvl = np.take(subWvl,idx[j])

            cas = np.take(subSpec,idx[j])

            if printSel:
                #blue to red: the order a dichroic chain would split them
                sel = [Exact_Line(w, s.split(' ')[0]) + (s,)
                       for w, s in zip(cwvl, cas)]
                sel.sort(key=lambda t: t[0])
                print(f'  stack {j+1}: {", ".join(stackL[j])}')
                for exW, exIon, exPrev, s in sel:
                    tag = ' (prev tok)' if exPrev else ''
                    print(f'     {s.split(" ")[0]:<3s} {exIon:<3s} '
                          f'{exW:9.3f} nm{tag}')

            Plot_Lines(cwvl,cas,None,a[j])
            drawn = Plot_Zeeman_Spectra(a[j], spectra, cwvl, window = window,
                                        height = specH, cmap = cmap)
            if showEst and len(estCen):
                #continue the colour cycle so estimates do not reuse the
                #colours already spent on the measured spectra
                Plot_Zeeman_Estimates(a[j], estCen, estLow, estHigh, estLbl,
                                      cwvl, window = window, height = specH,
                                      cmap = cmap, cStart = len(drawn))

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
    filename = 'sparc_line_ids_widths_v1_balmer.xlsx'
    zFolder = 'broad_all_fixed'
    zFolder = 'split_ext_field_50_terms_h5'
    zFolder = 'broad_split_ext_field_all_calculable'


    zFile = zFolder + '/wavelengths.csv'
    #new name again so the previous run (50-term h5 + v0 line list) is kept
    
    stackL =[ ['B','He','O'],['N','C','B','W']]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              [ 'He', 'Ne', 'Ar', 'Kr'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],\
                ]
    stackL = [['O', 'N', 'C', 'B', 'He'],\
              ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','C'],\
              ['He', 'Ni', 'Mo', 'Al', 'C', 'N' ],]
    
    saveF = 'TestStack_v1_Fix_Fe685_Al360_Mo386_Kr469_Ar696_W498_Ni547_Cu521.npz'
    stackL = [['O', 'N', 'C', 'B', 'He'],\
            ['W', 'Mo', 'Fe', 'Ni', 'Cu', 'Al','B'],\
            ['He', 'Ni', 'Mo', 'Al', 'Fe' ,'W'],\
            ['He', 'Ne','Ar', 'Kr']]

    
    #saveF = 'NobleStack.npz'
    #stackL = [['He', 'Ne','Ar', 'Kr']]
    
    #regenerate wavelengths.csv (low/high/center of each Zeeman-split peak)
    #from the fixed term-grouped h5 dataset
    
    Load_H5_Zeeman(folder=zFolder,wvlD = 0.01,plot = False)


    wvl, spec, ion, prevTok = Load_data(filename)
    outWvl, outAS, outPT = Filter_Stacks(stackL,wvl, spec, ion, prevTok, force = { 'Fe': 685.482, 'Al': 360.193, 'Mo': 386.410, 'Kr': 469.365, 'Ar': 696.543, 'W': 498.259, 'Ni': 547.691, 'Cu': 521.820})

    #zFolder+filename -> Zeeman catalog = measured h5 envelopes + estimated
    #envelopes for the lines Curt has not computed yet
    scores = Grade_Stack(stackL, outWvl, outAS, outPT,zFile,
                         zFolder = zFolder, xlsxFile = filename)
    Save_GradedStack(saveF, outWvl,outAS,outPT,scores)
    
    
    outWvl, outAS, outPT, scores = Load_GradedStack(saveF)

    Plot_Stack(scores, stackL, outWvl, outAS, outPT,prevTokW = 0.5,uvSW = 2,zW = 1,zFilename = zFile)

    #Plot_lines(wvl, spec, ion)
