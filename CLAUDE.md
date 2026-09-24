# Filterscopes - SPARC filterscope line and filter selection

Aaron Rosenthal (CFS WAVE DIAG). Python tools for choosing emission lines to
measure on SPARC filterscopes, grading multi-line "stacks" of them, and
specifying the dichroic / band-pass filters that separate them. Zeeman data
(h5 term files) comes from Curt, a colleague.
Current state of work and open items: see HANDOFF.md.

## Environment
- Windows. Use the project venv: `filterscopes\Scripts\python.exe`
  (activate: `filterscopes\Scripts\activate`), Python 3.12.8.
- Pins that matter: numpy 2.1.2, pandas 2.2.3, h5py 3.15.1, scipy 1.14.1,
  matplotlib 3.10.0, openpyxl 3.1.5, drjit 1.1.0 (CUDA; only the legacy
  Grade_Zeeman_Grid uses it). requirements.txt is UTF-16 encoded.
- Line endings: repo stores LF, Windows working files are CRLF;
  `.gitattributes` (`* text=auto`) keeps diffs clean on any machine.
- `.gitignore` covers `*.npz` (~150 GB of graded stacks), `__pycache__/`,
  `~$*` and the `filterscopes/` venv, but the root still holds many untracked
  data/CAD files (h5 folders, .zip, .stp, .obj). Never `git add .` /
  `git add -A`; add files by name.

## Pipeline (StackMaker.py)
1. `Load_data(xlsx)` - selectable lines; >= 1 nm apart per element,
   previously-fielded (Prev_Tok) lines preferred.
2. `Filter_Stacks(stackL, ..., force={...})` - every one-line-per-species
   combination as float16 arrays (species x combos). `force` pins lines first.
3. `Merge_Zeeman_Catalog(zFolder, xlsx)` - measured envelopes
   (`<zFolder>/wavelengths.csv`, from Load_H5_Zeeman/LowHighCent at 1% of peak)
   plus `Est_Zeeman_Width` estimates for lines with no h5 file.
4. `Grade_Stack` - scores per combination: [wvl, prevTok, uv, zeeman], float16
   (nCombos, 4). `Save_GradedStack` -> .npz.
5. `Plot_Stack(..., nTop=, selFile=, plot=)` - figures, and the exact selected
   lines to `<run>_selected.csv`.
6. `FilterSpec.py <selected.csv>` - dichroic + band-pass specs (xlsx + PNG).
- `SpeciesLines.py` - per-species line survey plot.
- Defaults live in StackMaker: `DEFAULT_XLSX` (sparc_line_ids_widths_v1_balmer.xlsx),
  `DEFAULT_ZFOLDER` (broad_split_ext_field_all_calculable). Plot_Stack and
  SpeciesLines use them - change them there, not per function.

## Commands (from this folder)
- `filterscopes\Scripts\python.exe StackMaker.py` - WARNING: __main__ regenerates
  wavelengths.csv, regrades everything and overwrites `saveF`. Slow, huge files.
- `filterscopes\Scripts\python.exe FilterSpec.py <run>_selected.csv [--rank N] [--stack 2 3] [--cwl-tol 0.1] [--aoi 3] [--chain red-first] [--save]`
- `filterscopes\Scripts\python.exe SpeciesLines.py C He O --save`
- Quick check without regrading: `Load_GradedStack(npz)` then
  `Plot_Stack(..., nTop=3, plot=False, selFile=...)`.

## Gotchas
- Combination wavelengths are float16 (0.25-0.5 nm steps). Recover exact lines
  with Plot_Stack's `Exact_Line` (accepts only within half a float16 step,
  falls back to the full spreadsheet). `Zeeman_Line_LUT` indexes by float16 bits
  per species, which relies on same-element lines being >= 1 nm apart.
- pandas: a column named `stack` must be read as `df['stack']`; `df.stack` is a
  DataFrame method and silently compares as False.
- Same-type quotes nested in f-strings only parse on Python >= 3.12.
- 'charge' in the spreadsheet and h5 files is the spectroscopic number (He I = 1).
- jk-coupled lines (Ar I, Ne I, ...) have no Lande g, so their estimated
  envelope is the conservative lambda^2-scaled maximum (C_MAX_JK): an upper bound.
- `Grade_Stack(force=)` sets excluded rows to -inf; Plot_Stack maps -inf*0 = NaN
  back to -inf so they can never rank first.
- Record AI-assisted changes in `ai_usage_log.md` (dated, newest first).

## Working with Aaron
- Explain the reasoning behind code changes and the technique used.
- Offer improvements or alternatives rather than simply agreeing.
- Back physics claims with citations; nuance is welcome.
- Strong in Python; explain git steps (and anything ssh/IT) explicitly.
