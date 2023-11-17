# Debugging PanDDA 2

## Recovering Failed Runs

If PanDDA 2 has begun processing datasets but ends before it completes, for example because it was cancelled by a cluster scheduler, then it is possible to continue from the last dataset that completed processing.

This is as simple as running the same PanDDA command that began the run again. The program will detect that processed datasets are present and then continue.

A warning with this feature is that if the input data has changed, new datasets will be processed against the current data, not the original data. This could happen if, for example, new datasets have been collected.

## Discovering Which Datasets Were Used to Characterize the Ground State

The information on which datasets were used to characterize the statistical model from which the Z-map, events and event maps can be helpful in determining whether sensible ground state models were used or explaining anomalies in Z-maps or mean maps. This information can be found in two places:

1. The Console Output: Under each dataset's subheading in the console output it a section for which datasets were used for each statistical model, and which statistical model was selected.

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                             Mpro-x0072 : 1 / 54                              │
╰──────────────────────────────────────────────────────────────────────────────╯
    Resolution: 1.65
    Processing Resolution: 1.65
───────────────────────────── Comparator Datasets ──────────────────────────────
    Mpro-x0014 Mpro-x0017 Mpro-x0019 Mpro-x0022 Mpro-x0025 Mpro-x0026 Mpro-x0027
    Mpro-x0030 Mpro-x0040 Mpro-x0041 Mpro-x0058 Mpro-x0059 Mpro-x0072 Mpro-x0074
    Mpro-x0090 Mpro-x0091 Mpro-x0103 Mpro-x0104 Mpro-x0105 Mpro-x0113 Mpro-x0114

...

────────────────────────────── Model Information ───────────────────────────────
    Processed Models: [10, 7, 9]
    Selected model: 10
    Model Number: 1
        Processed: False
        Mpro-x2041 Mpro-x2110 Mpro-x2139 Mpro-x2903 Mpro-x2967 Mpro-x2976
        Mpro-x2996 Mpro-x3002 Mpro-x3004 Mpro-x3015 Mpro-x3037 Mpro-x3071
        Mpro-x3072 Mpro-x3073 Mpro-x3102 Mpro-x3104 Mpro-x3105 Mpro-x3118
        Mpro-x3121 Mpro-x3129 Mpro-x3145 Mpro-x3151 Mpro-x3173 Mpro-x3215
        Mpro-x3255
...
    Model Number: 7
        Processed: True
        Mpro-x0025 Mpro-x0103 Mpro-x0153 Mpro-x0222 Mpro-x0227 Mpro-x0242
        Mpro-x0263 Mpro-x0271 Mpro-x0278 Mpro-x0295 Mpro-x0302 Mpro-x0313
        Mpro-x0340 Mpro-x0343 Mpro-x0392 Mpro-x0397 Mpro-x0480 Mpro-x0523
        Mpro-x0997 Mpro-x1031 Mpro-x1151 Mpro-x1245 Mpro-x1266 Mpro-x1288
        Mpro-x2150
...
    Model Number: 10
        Processed: True
        Mpro-x0014 Mpro-x0017 Mpro-x0019 Mpro-x0022 Mpro-x0025 Mpro-x0026
        Mpro-x0027 Mpro-x0030 Mpro-x0040 Mpro-x0041 Mpro-x0058 Mpro-x0059
        Mpro-x0072 Mpro-x0074 Mpro-x0090 Mpro-x0091 Mpro-x0103 Mpro-x0104
        Mpro-x0105 Mpro-x0113 Mpro-x0114 Mpro-x0150 Mpro-x0153 Mpro-x0174
        Mpro-x0221 Mpro-x0222 Mpro-x0227 Mpro-x0231 Mpro-x0240 Mpro-x0242
─────────────────────────── Processed Model Results ────────────────────────────
    Model Number: 10
        Number of events: 5
    Model Number: 7
        Number of events: 2
    Model Number: 9
        Number of events: 3


```
2. The Processed Dataset yaml: Located in <pandda output directory>/processed_datasets/<dataset name>/processed_dataset.yaml, this file contains information on which datasets were used and which model was selected. This file also contains additional information such as the score of the model and details of the events.

```yaml
Summary:
  Processing Resolution: 1.65
  Comparator Datasets:
  - Mpro-x0014
  - Mpro-x0017
  - Mpro-x0019
...
  Selected Model: 10
  Selected Model Events:
  - 1
  - 2
Models:
  1:
    Processed?: false
    Characterization Datasets:
    - Mpro-x2041
    - Mpro-x3072
    - ...
    - Mpro-x2903
    - Mpro-x3118
    Model Score: 0.46
    Events: {}
...
  7:
    Processed?: true
    Characterization Datasets:
    - Mpro-x0392
    - Mpro-x0103
...
    - Mpro-x0271
    - Mpro-x0278
    Model Score: 0.26
    Events:
      1:
        Score: 0.77762371301651
        BDC: 0.95
        Size: 5.216026807289329
        Centroid:
        - 13.797403329950091
        - 31.124850986842112
        - 5.65212530923325
        score: 0.593933344202621
        Local Strength: 24.15843994955058
        Build Score: -175.38088989257812
        Noise: 1118.583251953125
        Signal: 664.3638916015625
        Num. Contacts: 12
        Num. Points: 357.0
        Optimal Contour: 5.358448997892515
        RSCC: 0.27030460363399444
        Ligand Centroid:
        - 13.431249381912282
        - 29.929987307326126
        - 5.593201194030271
...

```