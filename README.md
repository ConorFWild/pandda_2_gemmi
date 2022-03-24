# PanDDA 2

## Reporting Errors

PanDDA 2 is still in development and feedback is always appreciated! 

If you have a problem with installation, then it is most likely specific to your system and best to just email me.

If the program errors while it runs, it is most helpful if you include the command line output and the json log in a github issue. 

If you uncertain about the correctness of the results, then a github issue is appropriate if you can share information publicly, in particular screenshots of maps or ligand fits. If you cannot, then an email to me is the best way to raise the concerns. Either way, please include the program output, the json log and screenshots of the offending z maps/event maps/autobuilds.

## Installation

It is reccomended that you install PanDDA 2 in it's own python 3.8 anaconda enviroment. This can be achieved by installing anaconda and then:

```bash
conda create -n pandda2 python=3.9
conda activate pandda2
conda install -c conda-forge -y fire numpy scipy joblib scikit-learn umap-learn bokeh dask dask-jobqueue hdbscan matplotlib 
conda install -c conda-forge -y seaborn
conda install -c conda-forge -y rdkit
pip install ray
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd pandda_2_gemmi
pip install .
cd _gemmi
pip install .

```

Installing PanDDA 2 this way will add various scripts to your path, but only while you are in this anaconda enviroment.



## Running

Once you have installed PanDDA 2 in a conda enviroment, running it is as simple as:

```bash
python /path/to/analyse.py <data directories> <output directory> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" <options>

```


### A minimal run

If you want to run the lightest possible PanDDA (no clustering of datasets, no autobuilding, ect: basically PanDDA 1), then a command like the following is appropriate:

```bash
python /path/to/analyse.py <data directories> <output directory> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --autobuild=False --rank_method="size" --comparison_strategy="high_res_random" <options>

```


### Running with autobuilding
PanDDA 2 supports the autobuilding of events and ranking them by autobuildability. All one needs to do is ensure that BUSTER is setup in their path (and hence ana_pdbmaps and rhofit) and run PanDDA 2 with the autobuild flag on.

An example:
```bash
python /path/to/analyse.py <data dirs> <output dirs> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' --autobuild=True <options>

```


### Running with distributed computing at Diamond

It is strongly reccomended that if you are qsub'ing a script that will run PanDDA 2 you set up your enviroment on the head node (by activating the anaconda enviroment in which PanDDA 2 is installed) and use the "-V" option on qsub to copy your current enviroment to the job.

An example of how to run with distributed computing at Diamond Light Source is as follows:
```bash
# Ensuring availability of Global Phasing code for autobuilding and phenix for building cifs
module load phenix
module load buster

# Put the following in the file submit.sh
python /path/to/analyse.py <data dirs> <output dirs> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' --global_processing="distributed" <options>

# Submitting
chmod 777 submit.sh
qsub -V -o submit.o -e submit.e -q medium.q -pe smp 20 -l m_mem_free=15G submit.sh

```


### Running with distributed computing from condor
```bash
python ./pandda_gemmi/analyse.py /data/share-2/conor/pandda/data/pandda_inputs/BRD1 /data/share-2/conor/pandda/output/pandda_2_BRD1 --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("FWT","PHWT")' --autobuild=True --global_processing="distributed" --distributed_scheduler="HTCONDOR" --local_cpus=20

```

