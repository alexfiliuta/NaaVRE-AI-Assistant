"""True isolated cell units: manually supplied inputs, exactly ONE generated cell."""
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_lab_reference import run_calculation_cell,reviewed_cells


@pytest.fixture(scope='module',autouse=True)
def reviewed_source():
    reviewed_cells('plants'); reviewed_cells('energy')


@pytest.mark.parametrize('value,flagged',[(39.999,False),(40,True),(40.001,True)])
def test_flag_cell_in_isolation(value,flagged):
    original = {'reading_date':'2026-01-01','building_id':'A','electricity_kwh':value}
    result = run_calculation_cell('energy',2,{'valid_records':[original]})
    assert bool(result['flagged_records'])==flagged
    assert 'flagged_high' not in original  # Does not mutate valid observations.


def typed_plant(day,height,plant='X'):
    return {'measurement_date':datetime.fromisoformat(day),'height_cm':height,
            'plant_id':plant,'treatment':'control'}


@pytest.mark.parametrize('reverse',[False,True])
def test_endpoint_cell_in_isolation(reverse):
    records = [typed_plant('2026-03-01',8),typed_plant('2026-03-22',12)]
    if reverse:
        records.reverse()
    result = run_calculation_cell('plants',3,{'parsed_records':records})
    change = result['plant_changes'][0]
    assert change['change_cm']==4 and change['n_measurements']==2
    assert change['first_date']==datetime(2026,3,1)
    assert change['last_date']==datetime(2026,3,22)


def test_endpoint_cell_excludes_single_measurement_in_isolation():
    result = run_calculation_cell('plants',3,{'parsed_records':[typed_plant('2026-03-01',8)]})
    assert not result['plant_changes'] and len(result['excluded_plants'])==1


def test_treatment_statistics_cell_in_isolation():
    records = [typed_plant('2026-03-01',10,'X'),typed_plant('2026-03-01',20,'Y')]
    changes = [{'treatment':'control','change_cm':2},{'treatment':'control','change_cm':4}]
    result = run_calculation_cell('plants',4,{'parsed_records':records,'plant_changes':changes})
    summary = result['treatment_summary'][0]
    assert summary['n_plants']==2
    assert summary['mean_height_cm']==summary['median_height_cm']==15
    assert summary['min_height_cm']==10 and summary['max_height_cm']==20
    assert result['treatment_change_summary']['control']==[2,4]


def test_energy_statistics_cell_in_isolation():
    records = [{'reading_date':'2026-01-01','building_id':'A','electricity_kwh':0.0},
               {'reading_date':'2026-01-01','building_id':'B','electricity_kwh':5.0}]
    result = run_calculation_cell('energy',3,{'valid_records':records})
    assert result['building_summaries']['A']['mean_kwh']==0
    assert result['building_summaries']['B']['total_kwh']==5
    assert result['overlap_comparison']==[{'reading_date':'2026-01-01','building_A_kwh':0.0,'building_B_kwh':5.0,'difference_kwh':-5.0}]
