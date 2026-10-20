"""Negative tests: the independent grader must detect wrong numeric answers."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_lab_reference import PLANT_ENDPOINTS,PLANT_GROUPS,grade_plants,canonical_date


def good_plant_result():
    return {'rows':24,'gains':[
        {'plant_id':pid,'treatment':t,'first_date':'2026-03-01','last_date':'2026-03-22',
         'first_height_cm':first,'last_height_cm':last,'height_gain_cm':gain,'n_measurements':4}
        for pid,(t,first,last,gain) in PLANT_ENDPOINTS.items()],
        'summaries':[
            {'treatment':t,'measurement_date':d,'n_plants':3,**dict(zip(
                ['mean_height_cm','median_height_cm','min_height_cm','max_height_cm'],v))}
            for (t,d),v in PLANT_GROUPS.items()],
        'treatment_mean_gain':{'control':5.6,'fertilized':9.9}}


def test_complete_independently_assembled_plant_answer_accepted():
    grade_plants(good_plant_result())


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(rows=23),
    lambda r:r['gains'][0].update(height_gain_cm=13.8),
    lambda r:r['gains'][0].update(first_date='2026-03-22'),
    lambda r:r['gains'][0].update(n_measurements=24),
    lambda r:r['gains'].append(copy.deepcopy(r['gains'][0])),
    lambda r:r['summaries'][0].update(n_plants=12),
    lambda r:r['summaries'][0].update(mean_height_cm=5.6),
    lambda r:r['summaries'].pop(),
    lambda r:r['treatment_mean_gain'].update(control=0),
])
def test_wrong_plant_answer_rejected(mutation):
    result = good_plant_result()
    mutation(result)
    with pytest.raises(AssertionError):
        grade_plants(result)


@pytest.mark.parametrize('label',['aggregated','2026-02-30','2026-03-01not-a-timestamp'])
def test_invented_or_invalid_date_label_rejected(label):
    with pytest.raises(ValueError):
        canonical_date(label)


@pytest.mark.parametrize('label',['2026-03-01','2026-03-01T00:00:00','2026-03-01T00:00:00Z'])
def test_valid_iso_dates_accepted(label):
    assert canonical_date(label)=='2026-03-01'
