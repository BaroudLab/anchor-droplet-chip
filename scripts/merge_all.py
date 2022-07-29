# import snakemake

from adc.merge import merge_all

merge_all(
    paths_day1=snakemake.input.day1,  # noqa: F821
    paths_day2=snakemake.input.day2,  # noqa: F821
    concentrations=snakemake.params.concentrations,  # noqa: F821
    threshold=snakemake.config["threshold"],  # noqa: F821
    table_path=snakemake.output.table,  # noqa: F821
    swarm_path=snakemake.output.swarm,  # noqa: F821
    prob_path=snakemake.output.prob,  # noqa: F821
    prob_log_path=snakemake.output.prob_log,  # noqa: F821
)
