# Representative study images

These are representative images actually used in the study, rather than synthetic schema examples. 

Folders are organised by evaluation setting (`Internal` or `External`), modality (`Brain MRI`, `Retina OCT`, or `Chest Xray`), and binary label (`0` for normal, `1` for disease). Within each label folder, `clean` contains clean views and artefact folders contain corrupted views where supplied. Source filenames identify corresponding views. Unseen artefacts are supplied under `new_artefacts` for internal samples.

Original dataset sources are listed in the root [README](../README.md#data-availability). Obtain full datasets from their providers under the applicable terms. These samples alone do not reconstruct the original study cohorts or source-image split assignments.

To run the scripts, prepare metadata with `dataset` (`mri`, `oct`, or `xray`), `filepath`, `fileindex`, `binarylabel`, `severity`, `corruption`, and `use_type`. All views of one source image must share its `fileindex` and split. Use `severity=clean` for clean images and `severity=weak` for corrupted views; preserve the artefact type in `corruption`. See the [root README](../README.md#quick-start) for setup and commands.
