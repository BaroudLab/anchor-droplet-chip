configfile: "config.yaml"

rule align:
    input:
        data="{concentration}.tif",
        template="template_bin16_bf.tif",
        mask="labels_bin2.tif",
    params:
        binnings="[2,16,2]"
    output:
        aligned="{concentration}-aligned.tif"
    shell:
        "python -m adc.align {input.data} {input.template} {input.mask} --binnings={params.binnings} --path_to_save={output}"

rule count:
    input:
        "{concentration}-aligned.tif"
    output:
        "{concentration}-aligned-count.csv"
    shell:
        "python -m adc.count {input} {output}"

def get_tables(day):
    return [f"{file[:-4]}-aligned-count.csv" for file in config[day]]

rule table:
    input:
        day1 = get_tables("day1"),
        day2 = get_tables("day2")
    params:
        concentrations = expand("{concentrations}", concentrations=config["concentrations"]),
    output:
        table="table.csv",
        swarm="table-swarm_plot.png",
        prob="table-prob_plot.png",
        prob_log="table-prob_plot_log.png",
    script:
        "scripts/merge_all.py"
