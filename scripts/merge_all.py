import snakemake

from adc.merge import merge_all

merge_all(
    paths_day1=snakemake.input.day1,
    paths_day2=snakemake.input.day2,
    concentrations=snakemake.params.concentrations,
    threshold=snakemake.config["threshold"],
    table_path=snakemake.output.table,
    swarm_path=snakemake.output.swarm,
    prob_path=snakemake.output.prob,
    prob_log_path=snakemake.output.prob_log,
)
