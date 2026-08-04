# Binding-contacts-analysis-in-ChimeraX-

# DockAnalyzer

DockAnalyzer is a modular Python toolkit for identifying, classifying, scoring, reporting, and visualizing protein–ligand interactions from molecular docking poses in UCSF ChimeraX.

## Main Features

* General atomic contact detection
* Hydrogen-bond analysis
* Hydrophobic interaction analysis
* π–π, cation–π, anion–π, and amide–π interactions
* Salt-bridge detection
* Residue- and pose-level scoring
* Multipose comparison and ranking
* Statistical summaries, reports, tables, and visual outputs

## Basic Workflow

1. Download and extract the DockAnalyzer package.
2. Open UCSF ChimeraX.
3. Open the prepared receptor structure.
4. Open all individual ligand poses in the same ChimeraX session.
5. Run `analyze.py`.
6. Review the generated reports, tables, statistics, and visualizations.

Ligand poses must be provided as individual structure files. When multiple poses are stored in a single `.pdbqt` file, use the provided Splitting Poses Code before running DockAnalyzer.

## Requirements

* UCSF ChimeraX
* A supported 64-bit operating system
* Prepared receptor and ligand-pose files
* Read and write permission for the input and output directories

A separate Python installation is not normally required when DockAnalyzer is executed inside ChimeraX.

## Project Structure

The toolkit is organized into specialized modules for configuration, geometry, contact detection, interaction classification, scoring, export, reporting, visualization, and workflow integration.

The main execution file is:

```text
analyze.py
```

## Generated Results

Depending on the selected configuration, DockAnalyzer may generate:

* Interaction tables: `.csv`, `.tsv`, or `.xlsx`
* Analysis reports: `.html`, `.md`, or `.txt`
* Structured results: `.json`
* Molecular snapshots: `.png`
* Pose rankings and residue summaries
* Analysis metadata and execution information

## Documentation

Project documentation is available through the DockAnalyzer GitHub Pages website, including:

* Project Description
* DockAnalyzer User Guide
* DockAnalyzer Download
* Splitting Poses Code
* Citations and License

## Citation

DockAnalyzer is currently unpublished and does not yet have a DOI.

Recommended citation:

> Bastos, L. M. (2026). *DockAnalyzer: A modular Python toolkit for protein–ligand interaction analysis in UCSF ChimeraX* [Computer software]. GitHub repository.

Citation metadata is also available in `CITATION.cff`.

## License

The DockAnalyzer source code is distributed under the MIT License.

Original documentation, website content, tutorials, explanatory text, and original figures are licensed under the Creative Commons Attribution 4.0 International License, unless otherwise stated.

See:

* `LICENSE`
* `LICENSE-DOCUMENTATION.md`
* `CITATION.cff`

## Research Use Notice

DockAnalyzer is intended as a research and analysis support tool. Its results should be interpreted together with docking affinity, structural inspection, experimental evidence, and appropriate validation procedures.

The software does not replace experimental validation or the independent assessment of molecular docking protocols.

## Author

**Leonardo Marensi Bastos**

2026
