# anchor-droplet-chip
Analyse achored droplets with fluorescent bacteria.

We are imaging the entire chip using 20x 0.7NA objective lens using automatic stitching in NIS.
Bright-field image 2D and TRITC-3D acquired. The 3D stack is converted to 2D using maximum projection in NIS or Fiji. Both channels are then merged together and saved as a tif stack. After that this package can be applied to detect the individual droplets and count the fluorescent cells.

As the chips are bonded to the coverslip manually, they contain a randon tilt and shift, so detecting individual droplets proved to be unreliable. The current approach consisnts of preparing a well-lebelled template bright-field image and a labelled mask and matching the experimental brightfield image to the template.
![Paper outline(1)](https://user-images.githubusercontent.com/11408456/178001287-513e6398-c4e0-4946-b38f-6cb98dc0ee6c.svg)

## Installation
```bash
pip install git+https://github.com/BaroudLab/anchor-droplet-chip.git
```
## Usage

1. Notebook: `jupyter lab example.ipynb`
2. Command line:

    `python -m adc.align --help`

    `python -m adc.count --help`

### Dowloading the raw data
Head to release page https://github.com/BaroudLab/anchor-droplet-chip/releases/tag/v0.0.1 and download files one by one.

Or

Execute the notebook example.ipynb - the data will be fetched automatically.

### Aligning the chips with the template and the mask

Day 1:
```bash
python -m adc.align day1/00ng_BF_TRITC_bin2.tif template_bin16_bf.tif labels_bin2.tif
```
This command will create the stack day1/00ng_BF_TRITC_bin2-aligned.tif, which can be viewed in Fiji.
![Screenshot of 00ng_BF_TRITC_bin2-aligned.tif](https://user-images.githubusercontent.com/11408456/176169270-3d494fc3-a771-4bf0-859e-c9cc853ce2d9.png)

Day 2:
```bash
python -m adc.align day2/00ng_BF_TRITC_bin2_24h.tif template_bin16_bf.tif labels_bin2.tif
```

### Counting the cells day 1 and day2
```
python -m adc.count day1/00ng_BF_TRITC_bin2-aligned.tif day1/counts.csv
python -m adc.count day2/00ng_BF_TRITC_bin2_24h-aligned.tif day2/counts.csv
```

### Combining the tables from 2 days
```
python adc.merge day1/counts.csv day2/counts.csv table.csv
```

### Plotting and fitting the probabilities


## Sample data

Batch processing:
`python -m adc.io 6940212 data/`
`snakemake -d data table.csv -c4`
