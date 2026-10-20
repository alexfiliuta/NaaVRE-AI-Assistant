# CSV quality procedure

Version 1. Applicability: tabular environmental observations.

Inspect the actual source schema before planning column operations. Preserve case and unit suffixes. Never assume that temperature means temperature_c or date means timestamp without evidence. Check required headers separately from missing row values. Preserve raw observations, report exclusions, and use safe forward-slash paths. In plots, handle constant values and a single timestamp without division by zero. Use actual earlier output variable names; a filename is not a Python variable. Save derived files only under the requested output directory and never copy a file onto itself.

This curated procedure records development lessons. It does not certify every generated workflow or authorize any tool.
