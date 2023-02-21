configfile: "config.yaml"

rule align:
    input:
        data="{concentration}.tif",
        # template="template_bin16_bf.tif",
        template = "v3/template_bin16_v3.tif",
        # mask="labels_bin2.tif",
        mask = "v3/labels_bin2_v3.tif"
    params:
        binnings="[2,16,2]",
        cx = 0,#config['constraints']["tx"][0],
        sx = 150,#config['constraints']["tx"][1],
        cy = 0,#config['constraints']["ty"][0],
        sy = 50#config['constraints']["ty"][1]
    output:
        aligned="{concentration}-aligned.tif"
    shell:
        "python -m adc.align {input.data} {input.template} {input.mask} --binnings={params.binnings} --path_to_save={output} --sx={params.sx} --cx={params.cx} --sy={params.sy} --cy={params.cy}"

rule count:
    input:
        "{concentration}-aligned.tif"
    output:
        "{concentration}-aligned-count.csv"
    shell:
        "python -m adc.count {input} {output}"

def get_tables(day):
    return [f"{file[:-4]}-aligned-count.csv" for file in config[day]]

def get_aligned_tifs(day):
    return [f"{file[:-4]}-aligned.tif" for file in config[day]]

rule table:
    input:
        day1 = get_tables("day1"),
        day2 = get_tables("day2"),
        zarr1 = directory("day1.zarr"),
        zarr2 = directory("day2.zarr"),
    params:
        concentrations = expand("{concentrations}", concentrations=config["concentrations"]),
    output:
        table="table.csv",
        swarm="table-swarm_plot.png",
        prob="table-prob_plot.png",
        prob_log="table-prob_plot_log.png",
    script:
        "scripts/merge_all.py"

rule zarr:
    input:
        day1 = get_aligned_tifs("day1"),
        day2 = get_aligned_tifs("day2")
    output:
        zarr1 = directory("day1.zarr"),
        zarr2 = directory("day2.zarr"),
    script:
        "scripts/zarr.py"
