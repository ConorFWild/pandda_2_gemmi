# PanDDA 2

## New in version 0.1.0

 - Improved README.md
 - Phenix dependency removed
 - PanDDA defaults to using present cifs with no rebuilding
 - Cleaner logs
 - Improved ranking for unbuildable events
 - Memory performance optimizations
 - Which filters to apply to datasets can be configured with --data_quality_filters and --reference_comparability_filters

## Reporting Errors

PanDDA 2 is still in development and feedback is always appreciated! 

If you have a problem with installation, then it is most likely specific to your system and best to just email me.

If the program errors while it runs, it is most helpful if you include the command line output and the json log in a GitHub issue. 

If you uncertain about the correctness of the results, then a GitHub issue is appropriate if you can share information publicly, in particular screenshots of maps or ligand fits. If you cannot, then an email to me is the best way to raise the concerns. Either way, please include the program output, the json log and screenshots of the offending z maps/event maps/autobuilds.


## Installation

It is recommended that you install PanDDA 2 in its own python 3.8 anaconda environment. If you do not have an Anaconda enviroment you can install one by following the instructions at https://www.anaconda.com/products/distribution#linux. 

Then:

```bash
conda create -n pandda2 python=3.9
conda activate pandda2
conda install -c conda-forge -y fire numpy scipy joblib scikit-learn umap-learn bokeh dask dask-jobqueue hdbscan matplotlib rich seaborn rdkit openbabel
pip install ray
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd pandda_2_gemmi
pip install -e .
cd _gemmi
pip install .
pip install numpy==1.21.0
```

Installing PanDDA 2 this way will add various scripts to your path, but only while you are in this anaconda environment.



## Running PanDDA 2

PanDDA 2 supports the autobuilding of events and ranking them by autobuildability. All one needs to do is ensure that BUSTER is set up in their path (and hence ana_pdbmaps and rhofit).


Once you have installed PanDDA 2 in a conda environment, it can be run from that enviroment with autobuilding and automated ground state identification with the following: 

```bash
python /path/to/analyse.py --data_dirs=<data directories> --out_dir=<output directory> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> <options>
```


### Minimal Run

If you want to run the lightest possible PanDDA (no clustering of datasets, no autobuilding, ect: basically PanDDA 1), then a command like the following is appropriate:

```bash
python /path/to/analyse.py --data_dirs=<data directories> --out_dir=<output directory> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> --autobuild=False --rank_method="size" --comparison_strategy="high_res_first" <options>
```


### Running With Distributed Computing At Diamond

It is strongly recommended that if you are qsub'ing a script that will run PanDDA 2 you set up your environment on the head node (by activating the anaconda environment in which PanDDA 2 is installed) and use the "-V" option on qsub to copy your current environment to the job.

An example of how to run with distributed computing at Diamond Light Source is as follows:
```bash
# Ensuring availability of Global Phasing code for autobuilding and phenix for building cifs
module load ccp4
module load buster

# Put the following in the file submit.sh
python /path/to/analyse.py --data_dirs=<data dirs> --out_dir=<output dirs> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> --global_processing="distributed" <options>

# Submitting
chmod 777 submit.sh
qsub -V -o submit.o -e submit.e -q medium.q -pe smp 12 -l m_mem_free=15G submit.sh
```

## PanDDA 2 Usage FAQ

### The event map doesn't resemble the protein

This is to be expected if the event map's event does not is not a fragment or actual meaningful change: in such cases the event map is effectively random and should be ignored.

If there is clearly a fragment present but the event map's quality seems low for the protein: this is typically due to a poorly characterized ground state model and such event maps can be used as normal.

### An event's density looks like the soaked fragment but it has not been autobuilt

There are two possibilities. The first is that autobuilding the event has failed, typically because the outlying Z-values form a small blob.

The second possibility is that the event has been autobuilt, but PanDDA prefers another event in the protein to merge into the model shown in pandda.inspect. This is because in the current version of PanDDA inspect only one autobuild can be chosen to display for any dataset, regarless of how many events it has. PanDDA does occasionally chose a poor autobuild over a good one, so it is important in such cases to manually build the good density and then check that if PanDDA has placed a fragment it too is in a sensible place.

### Many datasets don't seem to have any event maps

It is important to check the PanDDA logs to make sure that datasets you are interested in have not been filtered from consideration. 

### An event has been autobuilt but incorrectly

Autobuilding PanDDA events is not perfect and the method can sometimes fail. If you see an event in pandda.inspect that appears to resemble the fragment but the autobuild is poor, the advice is to correct the autobuild and save the new model.

## How PanDDA 2 works

PanDDA 2 differs from PanDDA 1 in two major methodological ways. 

Firstly, it attempts to identify which sets of datasets should be used to produce a statistical model which has the optimal contrast (to each test dataset individually). This allows it to handle subtle heterogeneity between datasets. 

Secondly it attempts to autobuild the events returned, and then rank the events based on the quality of the model of the fragment that could be constructed. This allows improved rankings of events.

### Statistical Model Dataset Selection

PanDDA 2 selects the datasets to construct each statistical model it tries by identifying the sets of datasets which are closest to each other. This is achieved by:
 - Finding all pairwise distances between electron density maps
 - Finding the nearest neighbours of each dataset's electron density map
 - Finding the non-overlapping neighbourhoods with the minimum standard deviation between the maps they contain.

### Statistical Model Selection

Currently PanDDA 2 only chooses one statistical model to progress to event map generation and autobuilding. This is done by fitting several conformations of the expected fragment into each event from each statistical by differential evolution. 

The statistical which has the best fragment fit is then selected for progression to event map generation and autobuilding.

### Autobuilding

PanDDA 2 autobuilds the events of each dataset. This is done with a custom Rhofit invocation to account for the differing distribution of values between event maps and normal crystallographic 2Fo-Fc maps.

### Autobuild Selection

Current limitations with the interaction between pandda.inspect and PanDDA 2 mean that it is only possible to show one autobuild in the GUI. 

Therefore, the highest scoring autobuild by RSCC from any event in each dataset is selected and included in the initial model pandda.inspect shows the user. 

This has the effect that users may open apparently good hit density with no autobuild present, if another hit which is better fit by the autobuilding is present in the same dataset.


