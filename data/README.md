# Data Directory

This directory is intentionally gitignored except for this README.

Download these Google Drive assets here:

```text
data/cropped_images/
data/Data_11.5.23_modified.csv
```

`cropped_images/` should contain one folder per patient `Study_ID`, with ultrasound images from:

- abdomen (`A`/`AB` filename patterns)
- biceps (`B`/`BICEP` filename patterns)
- quadriceps (`Q`/`QUAD` filename patterns)

The label CSV should include ADP/Pea Pod reference body-composition labels:

```text
Study_ID,FM,FFM,Weight_visit,Length_visit
```

Do not commit clinical images, labels, or derived patient-level outputs.
