"""
Per-species line survey.

Plots every candidate measurement line for one species (e.g. 'C') as a vertical
line, over the Zeeman spectra of everything lying within a band around it, so
you can see at a glance which of that species' lines are clean, which are
crowded, and which have already been fielded on a tokamak.

Built on StackMaker so the spectra, envelopes, colours and labels match what
Plot_Stack shows for a graded stack.

    python SpeciesLines.py C
    python SpeciesLines.py C He O --window 4 --save

or from python:

    from SpeciesLines import Plot_Species
    Plot_Species('C')
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from StackMaker import (Load_data, Load_H5_Spectra, Plot_Zeeman_Spectra,
                        Plot_Zeeman_Estimates, Merge_Zeeman_Catalog,
                        Zeeman_Line_Scores, wave_to_rgb)

XLSX = 'sparc_line_ids_widths_v0.xlsx'
XLSX = 'sparc_line_ids_widths_v1_balmer.xlsx'
ZFOLDER = 'split_ext_field_50_terms_h5'


def Cluster_Lines(wvl, gap):
    """
    Group lines that sit close enough to share a zoom panel.

    Inputs:
    - wvl: nparray - line wavelengths in nm (any order)
    - gap: float - start a new group when the jump exceeds this (nm)
    Outputs:
    - groups: list of nparray - the wavelengths in each group, left to right
    """
    w = np.sort(np.asarray(wvl, dtype=float))
    if w.size == 0:
        return []
    cut = np.flatnonzero(np.diff(w) > gap) + 1
    return np.split(w, cut)


def Load_Species_Lines(species, xlsxFile=XLSX, zFolder=ZFOLDER, downselect=True,
                       zMode='envelope'):
    """
    Everything needed to draw one species: its candidate lines, whether each was
    previously fielded, and how blended each is under the Zeeman catalog.

    Inputs:
    - species: str - element symbol, e.g. 'C'
    - xlsxFile: str - candidate line list
    - zFolder: str - folder of h5 term files
    - downselect: bool - use the selectable lines Load_data produces (lines
      within 1 nm of each other collapsed to one, which is what the stack
      optimiser actually chooses from). False uses every spreadsheet row.
    - zMode: 'envelope' or 'center', passed to Zeeman_Line_Scores
    Outputs:
    - dict with wvl, ion, prevTok, zScore and the catalog arrays
    """
    if downselect:
        wvl, spec, ion, prevTok = Load_data(xlsxFile)
    else:
        from StackMaker import Load_xlsx
        wvl, spec, _, ion, prevTok, _, _ = Load_xlsx(xlsxFile)

    m = (spec == species)
    if not m.any():
        raise ValueError(f"no lines for species {species!r}; "
                         f"available: {', '.join(sorted(set(spec)))}")

    cen, low, high, isEst = Merge_Zeeman_Catalog(zFolder, xlsxFile, verbose=False)
    zScore, zCount = Zeeman_Line_Scores(wvl[m], cen, low, high, mode=zMode)

    order = np.argsort(wvl[m])
    return dict(wvl=wvl[m][order], ion=ion[m][order], prevTok=prevTok[m][order],
                zScore=zScore[order], zCount=zCount[order],
                cen=cen, low=low, high=high, isEst=isEst)


def _estimated_entries(zFolder, xlsxFile):
    """Estimated envelopes + display labels, from the catalog CSV."""
    Merge_Zeeman_Catalog(zFolder, xlsxFile, verbose=False)
    cat = pd.read_csv(os.path.join(zFolder, 'zeeman_catalog.csv'))
    est = cat[cat['estimated']]
    lbl = []
    for f_ in est['file']:
        p = str(f_).split('_')
        try:
            lbl.append(f'{p[0]} {p[1]} {float(p[2]):.1f} est')
        except (IndexError, ValueError):
            lbl.append(str(f_))
    return (est['center'].to_numpy(), est['low'].to_numpy(),
            est['high'].to_numpy(), lbl)


def Plot_Species(species, xlsxFile=XLSX, zFolder=ZFOLDER, window=3.0,
                 specH=3.0, cmap='hsv', downselect=True, showEst=True,
                 zMode='envelope', savePath=None, show=True):
    """
    Plot every candidate line of one species over the nearby Zeeman spectra.

    Layout is an overview strip across the species' whole range, then one zoom
    panel per cluster of lines. The zoom panels are the point: a species can
    span 300 nm while its Zeeman structure is a few tenths of a nm wide, so a
    single axis would show nothing useful.

    Vertical lines mark the candidate wavelengths, coloured by their visible
    emission colour (wave_to_rgb) exactly as Plot_Lines does inside Plot_Stack.
    A line previously fielded on a tokamak gets a star and a boxed label.

    Inputs:
    - species: str - element symbol, e.g. 'C'
    - xlsxFile, zFolder: str - line list and h5 term files
    - window: float - nm either side of each line to draw spectra for
    - specH: float - plot height of a fully normalised spectrum
    - cmap: str - cycling colour wheel for neighbouring spectra
    - downselect: bool - see Load_Species_Lines
    - showEst: bool - also draw estimated envelopes for lines with no h5 data
    - zMode: 'envelope' (conservative) or 'center'
    - savePath: str or None - write the figure here
    - show: bool - call plt.show()
    Outputs:
    - fig, axes
    """
    d = Load_Species_Lines(species, xlsxFile, zFolder, downselect, zMode)
    wvl, ion, prevTok, zScore = d['wvl'], d['ion'], d['prevTok'], d['zScore']

    spectra = Load_H5_Spectra(zFolder)
    if showEst:
        eCen, eLow, eHigh, eLbl = _estimated_entries(zFolder, xlsxFile)
    else:
        eCen = eLow = eHigh = np.array([]); eLbl = []

    groups = Cluster_Lines(wvl, gap=4*window)
    nG = len(groups)

    fig = plt.figure(figsize=(4.6*max(nG, 3), 7.6))
    gs = GridSpec(2, nG, figure=fig, height_ratios=[1, 3.1], hspace=0.35,
                  wspace=0.16)

    # ---------------- overview strip ----------------
    axO = fig.add_subplot(gs[0, :])
    for w, io, pt in zip(wvl, ion, prevTok):
        c = wave_to_rgb(w)
        axO.vlines(w, 0, 1, color=c, lw=1.6)
        if pt:
            axO.plot(w, 1.06, marker='*', ms=9, color='k', zorder=6)
    axO.set_ylim(0, 1.3); axO.set_yticks([])
    lo, hi = wvl.min(), wvl.max()
    pad = max(0.03*(hi - lo), 5)
    axO.set_xlim(lo - pad, hi + pad)
    axO.set_xlabel('Wavelength (nm)', fontsize=9)
    axO.grid(axis='x', color='0.9', lw=0.5)
    nUsed = int(prevTok.sum())
    nClean = int((zScore == 0).sum())
    axO.set_title(f'{species}: {len(wvl)} candidate lines  |  {nUsed} previously '
                  f'fielded  |  {nClean} Zeeman-clean ({zMode} test)',
                  fontsize=12, fontweight='bold')
    #show where each zoom panel looks
    for k, g in enumerate(groups):
        axO.axvspan(g.min() - window, g.max() + window, color='0.85',
                    alpha=0.55, lw=0, zorder=0)
        axO.text(g.mean(), 1.18, f'{k+1}', ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec='0.4'))

    # ---------------- zoom panels ----------------
    axes = []
    for k, g in enumerate(groups):
        ax = fig.add_subplot(gs[1, k])
        axes.append(ax)

        drawn = Plot_Zeeman_Spectra(ax, spectra, g, window=window,
                                    height=specH, cmap=cmap)
        if showEst and len(eCen):
            Plot_Zeeman_Estimates(ax, eCen, eLow, eHigh, eLbl, g, window=window,
                                  height=specH, cmap=cmap, cStart=len(drawn))

        #the species' own candidate lines on top. Stagger the labels over
        #several rows: a cluster can hold half a dozen lines within a few nm,
        #and side-by-side boxes overlap into an unreadable smear.
        nRow = 1 if len(g) == 1 else (2 if len(g) <= 4 else 3)
        for j, w in enumerate(g):
            i = int(np.argmin(np.abs(wvl - w)))
            c = wave_to_rgb(w)
            row = j % nRow
            yTop = specH*(2.20 + 0.62*row)
            ax.vlines(w, 0, yTop, color=c, lw=2.0, zorder=8)
            used = bool(prevTok[i])
            if used:
                ax.plot(w, yTop + specH*0.07, marker='*', ms=11, color='k', zorder=9)
            txt = f'{species} {ion[i]}\n{w:.3f}'
            if zScore[i] != 0:
                txt += f'\nblend {int(-zScore[i])}'
            ax.text(w, yTop + specH*0.15, txt, ha='center', va='bottom', fontsize=7.5,
                    color='k', zorder=9, fontweight='bold' if used else 'normal',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='#fff3cd' if used else 'white',
                              ec=c, lw=1.0, alpha=0.95))

        ax.set_xlim(g.min() - window, g.max() + window)
        ax.set_ylim(0, specH*(2.20 + 0.62*(nRow-1) + 1.05))
        ax.set_yticks([])
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.grid(axis='x', color='0.9', lw=0.5)
        ax.set_title(f'{k+1}', fontsize=9, loc='left', color='0.35')

    axes[0].set_ylabel('Zeeman spectra (normalised)', fontsize=9)

    leg = [Line2D([0], [0], color='0.3', lw=2, label=f'{species} candidate line'),
           Line2D([0], [0], marker='*', color='k', lw=0, ms=11,
                  label='used on a previous tokamak'),
           Line2D([0], [0], color='0.55', lw=1.2, label='measured Zeeman spectrum'),
           Patch(facecolor='0.8', edgecolor='0.4', ls='--', hatch='///',
                 label='estimated envelope (no h5 data)')]
    fig.legend(handles=leg, ncol=4, fontsize=9, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    if savePath:
        fig.savefig(savePath, dpi=140, bbox_inches='tight')
        print('saved', savePath)
    if show:
        plt.show()
    return fig, axes


def Species_Table(species, xlsxFile=XLSX, zFolder=ZFOLDER, downselect=True,
                  zMode='envelope'):
    """Print a short per-line summary for one species."""
    d = Load_Species_Lines(species, xlsxFile, zFolder, downselect, zMode)
    print(f"\n{species}: {len(d['wvl'])} candidate lines "
          f"({int(d['prevTok'].sum())} previously fielded)")
    print(f"  {'ion':>4s} {'wave (nm)':>10s} {'prev tok':>9s} {'blended with':>13s}")
    for w, io, pt, zs in zip(d['wvl'], d['ion'], d['prevTok'], d['zScore']):
        print(f"  {io:>4s} {w:10.3f} {('Y' if pt else '-'):>9s} "
              f"{int(-zs):>13d}")
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('species', nargs='+', help="element symbol(s), e.g. C He O")
    ap.add_argument('--window', type=float, default=3.0,
                    help='nm either side of each line to draw spectra for')
    ap.add_argument('--xlsx', default=XLSX)
    ap.add_argument('--zfolder', default=ZFOLDER)
    ap.add_argument('--mode', default='envelope', choices=['envelope', 'center'])
    ap.add_argument('--all-lines', action='store_true',
                    help='include lines removed by the 1 nm downselect')
    ap.add_argument('--no-est', action='store_true',
                    help='hide estimated envelopes')
    ap.add_argument('--save', action='store_true',
                    help='write species_<X>_lines.png instead of showing')
    a = ap.parse_args()

    for sp in a.species:
        Species_Table(sp, a.xlsx, a.zfolder, not a.all_lines, a.mode)
        Plot_Species(sp, xlsxFile=a.xlsx, zFolder=a.zfolder, window=a.window,
                     downselect=not a.all_lines, showEst=not a.no_est,
                     zMode=a.mode,
                     savePath=f'species_{sp}_lines.png' if a.save else None,
                     show=not a.save)
