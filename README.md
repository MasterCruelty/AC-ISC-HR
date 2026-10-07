# AC-ISC-HR
Project for exam of Affective Computing at University of Milan. Detecting inter-subject heart-rate synchronization (ISC-HR) during narrative stimuli. Based on Pérez et al. (2021) and reproduced on BBBD dataset(2026). 

## Inter-Subject Correlation of Heart Rate and Attention Modulation

The project reproduces two results reported by Pérez et al. (2021) on Experiment 2 of the BBBD
dataset. <br>
Methodology and results are in the final report.
 
## Requirements
 
```
pip install -r requirements.txt
```

## Data
 
The folder `data/` contains the ECG recordings of Experiment 2 of the BBBD dataset.<br>
`EXPERIMENT_ROOT` in `main.py` points to `data/` by default.
 
## Usage

```
python main.py
```
 
It is possible to choose the stages to run in `STAGES_TO_RUN` (inside `main.py`). By default all stages are executed.

## Output
 

- **Stage 1** : summary of one subject ECG recording.
- **Stage 2** : ISC-HR, p-value and significance of each subject for one
  (session, stimulus) combination.
- **Stage 3** : summary over all 10 combinations: subjects analysed, mean
  ISC-HR, number of significant subjects (R1).
- **Stage 4** : attention effect on synchrony (R2): significant subjects
  per stimulus, paired test per stimulus and ANOVA.

  
## References
 
- P. Pérez, J. Madsen, L. Banellis, B. Türker, F. Raimondo, V. Perlbarg,
  M. Valente, M.-C. Niérat, L. Puybasset, L. Naccache, T. Similowski,
  D. Cruse, L. C. Parra, J. D. Sitt. *Conscious processing of narrative
  stimuli synchronizes heart rate between individuals.* Cell Reports,
  36(11):109692, 2021. <https://doi.org/10.1016/j.celrep.2021.109692>
- J. Madsen, N. Kuppa, L. C. Parra. *The Brain, Body, and Behavior Dataset
  (BBBD): Multimodal Recordings during Educational Videos.* Scientific Data,
  13:920, 2026. <br> <https://doi.org/10.1038/s41597-026-07215-1> <br> <https://bbbd.pythonanywhere.com/>






