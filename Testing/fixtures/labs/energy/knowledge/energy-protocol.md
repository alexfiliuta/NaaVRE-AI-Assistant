# Building electricity demonstration protocol

This is synthetic teaching data authored for this prototype on 2026-10-07. It is not measured data, a scientific paper, or a dataset downloaded from NaaVRE. The values are deliberately fixed, not randomly generated, so checks are reproducible.

One row is one building's daily electricity use. The observation key is the pair `reading_date` and `building_id`. `reading_date` is an ISO calendar date without a time zone. `building_id` is text. `electricity_kwh` is daily energy in kilowatt-hours, not cumulative meter readings or power in kilowatts.

There are 14 dates, 2026-01-01 through 2026-01-14, for buildings A and B. A blank electricity value means missing, not zero. Reject invalid dates, duplicate observation keys and non-finite or negative numeric readings in a reported validation step. Do not silently impute or overwrite source values. Report the number of available readings when calculating means and sums.

For this demonstration only, flag a valid daily reading greater than or equal to 40 kWh for review. This is a supplied teaching rule, not a universal scientific threshold or proof that a meter is faulty. Retain valid flagged readings in the main descriptive comparison. Do not invent a statistical outlier definition or silently exclude the high value.

Compare daily trends and available-reading summaries for the two buildings. Missing days make full-period sums incomplete. A fair same-date total comparison can use the dates with a valid reading from both buildings and report the overlap count. Neither floor areas nor occupancy nor weather are supplied: do not infer energy efficiency or a cause for differences.

Draw a labelled daily line chart in kWh. Show a gap at a missing reading, rather than a zero or a continuous interpolated line. Distinguish buildings and mark the high reading. The notebook can use Python's standard library and IPython's SVG display. Save derived CSVs and the SVG under this lab's outputs directory. Keep the input CSV unchanged and create a useful sequence of focused, individually executable cells, without a fixed cell count.
