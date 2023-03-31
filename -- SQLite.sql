-- SQLite
SELECT datasets.id,
datasets.date,
datasets.antibiotic_type,
chips.chip_id,
chips.concentration,
datasets.unit,
datasets.path
FROM datasets
JOIN chips
ON chips.dataset_id = datasets.id
