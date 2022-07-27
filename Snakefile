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
