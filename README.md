# Condensed-phase molecular representation to link structure and thermodynamics in molecular dynamics
### Code accompanying the publication at <http://insert_doi_link.com>

The scripts were created specifically for this project, with specific expectations for directory paths 
and some hard-coded filenames and other settings.
* Python version >= 3.9
* package list `requirements.txt`

## Inserting angles and constraints into the GROMACS topology files for small molecule graphs:
* `isomorphism_find_unique.py`
    * A small number of structures make up the entire set, the variations between the small molecules 
    are based on different combinations of coarse-grained bead types.
* `make_template_file.py`
    * Suitable angles and constraints are manually optimized for one example of each of the recurring 
    structures and subsequently used as a template.
* `isomorphs_insert_constraints.py`
    * Iterating over the entire set of small molecule graphs, identifiying the correct template and inserting 
    the settings for angles and constraints.

## Generating the SLATM representations:
The Spectrum of London and Axilrod-Teller-Muto (SLATM) potential was defined by Huang, Symonds and von Lilienfeld, <https://arxiv.org/abs/1807.04259>.
* `analyze_structures.py`
    * Handles loading of trajectory files and the required steps to translate the coarse-grained MD trajectories 
    into SLATM representations.
    * `clean_trajectories.py`
        * Corrects for periodic boundary conditions, centers systems around the solutes, selects frames by solute 
        position.
    * `preprocessing.py`
        * Selects solutes and environment particles within the long-range interaction cutoff distance around the 
        solutes' center of mass.
    * `generate_representations.py`
        * Handles the generation of the list of possible many-body interactions and unique particle identifiers 
        required by the QML SLATM method.
        * Generates the SLATM representations.
        * Saves the results as pickled pandas dataframes.

## Analyzing a set of SLATM representations with PCA:
* `analyze_SLATMs.py`
    * Loads the SLATM representations and required additional files, 
    * Handles preprocessing of SLATM representations including normalization for PCA
    * PCA embedding of the SLATM representations

## Visualizing the PCA results:
* `generate_labels.py`
    * Loads principal components and additional information, generates descriptors for further analysis.
* `plot_cross_correlations.py`
    * Cross-correlates a descriptor to principal components, performs linear regression.
* `correlate_loadings_interactions.py`
    * Visualizes the most relevant interactions selected by their loading values (`eigenvecor * sqrt(eigenvalue)`) 
    to provide an idea about 3D structural aspects.
* `biplot_scores_weights.py`
    * Plots most pairs of principal components and their most relevant eigenvector coefficients. Colored by a 
    previously generated descriptor.
