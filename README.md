# anchor-droplet-chip
Analyse achored droplets with fluorescent bacteria

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
python -m adc.align 00ng_BF_TRITC_bin2.tif template_bin16_bf.tif labels_bin2.tif 
```
This command will create the stack 00ng_BF_TRITC_bin2-aligned.tif, which can be viewed in Fiji.
![Screenshot of 00ng_BF_TRITC_bin2-aligned.tif](https://user-images.githubusercontent.com/11408456/176169270-3d494fc3-a771-4bf0-859e-c9cc853ce2d9.png)

Day 2:
```bash
python -m adc.align 00ng_BF_TRITC_bin2_24h.tif template_bin16_bf.tif labels_bin2.tif 
```

### Counting the cells day 1 and day2



### Combining the tables from 2 days

### Pltting and fitting probabilities


## Sample data 

Check the releases section: 6 raw tif files from day1 and 6 raw tif files from day 2 are available as well as their aligned versions and corresponding tables.

