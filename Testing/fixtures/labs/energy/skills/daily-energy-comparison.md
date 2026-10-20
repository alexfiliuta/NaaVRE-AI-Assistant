# Daily building-energy comparison

Version 1. Applicability: daily per-building energy observations, not cumulative meters.

Inspect actual file headers before planning. Check the unique building/date observation key. Parse calendar dates and finite non-negative energy values. Treat blank readings as missing, never as zero. Report exclusions and per-building denominators. Apply the selected lab protocol's supplied high-reading rule and retain flagged valid values in summaries. When comparing total use on a common period, match dates across buildings and report coverage. Show missing dates as gaps in plots. Preserve source data and save derived results separately. Use one complete job per cell; the request determines the number of cells. Do not make causal or efficiency claims without the necessary supporting variables.
