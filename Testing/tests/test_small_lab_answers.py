"""Numeric unit/regression tests of the ORIGINAL model-generated calculation cells.

Known weaknesses are strict xfails: visible, reproducible and NOT counted as passes.
No LLM calls. Notebook hash review gate prevents executing newly edited code.
"""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_lab_reference import (PLANT_ENDPOINTS, PLANT_GROUPS, ENERGY_VALUES,
    calculation_cells, energy_expected, extract_original, grade_energy, grade_plants,
    number, read_rows, reviewed_cells)


@pytest.fixture(scope='module', autouse=True)
def reviewed_source_gate():
    # Must fail OUTSIDE xfail test bodies if notebook source changes.
    for kind in ['plants','energy']:
        reviewed_cells(kind)


def erow(value='20', day='2026-01-01', building='A'):
    return {'reading_date':day,'building_id':building,'electricity_kwh':value}


def prow(height='8', day='2026-03-01', plant='X', treatment='control', **extra):
    return {'measurement_date':day,'plant_id':plant,'treatment':treatment,
            'height_cm':height,'leaf_count':'4','biomass_g':'2',**extra}


@pytest.mark.parametrize('kind',['plants','energy'])
def test_actual_generated_calculations_match_independent_fixed_answers(kind):
    rows = read_rows(kind)
    result = extract_original(kind, calculation_cells(kind,rows))
    if kind=='plants':
        grade_plants(result,tolerance=5.1e-5)  # Original stores rounded 4-decimal means.
    else:
        grade_energy(result)


def test_plant_literal_answers_checked_directly_against_decimal_csv_calculation():
    rows = read_rows('plants')
    assert len(rows)==24
    for pid,(treatment,first,last,gain) in PLANT_ENDPOINTS.items():
        records = sorted([r for r in rows if r['plant_id']==pid],key=lambda r:r['measurement_date'])
        assert len(records)==4 and {r['treatment'] for r in records}=={treatment}
        assert Decimal(records[-1]['height_cm'])-Decimal(records[0]['height_cm'])==Decimal(str(gain))
        assert float(records[0]['height_cm'])==first and float(records[-1]['height_cm'])==last
    for (treatment,day),expected in PLANT_GROUPS.items():
        group = [r for r in rows if r['treatment']==treatment and r['measurement_date']==day]
        heights = sorted(Decimal(r['height_cm']) for r in group)
        assert len({r['plant_id'] for r in group})==3
        actual = [sum(heights)/len(heights),heights[1],min(heights),max(heights)]
        for a,e in zip(actual,expected):
            number(a,e,'independent Decimal group')


def test_energy_literal_answers_checked_directly_against_csv():
    rows = read_rows('energy')
    assert len(rows)==28
    for building,expected in ENERGY_VALUES.items():
        actual = sorted([r for r in rows if r['building_id']==building],key=lambda r:r['reading_date'])
        assert len(actual)==14
        assert [None if not r['electricity_kwh'].strip() else int(r['electricity_kwh']) for r in actual]==expected
    expected = energy_expected()
    assert expected['summaries']['A']['total_kwh']==301
    assert expected['summaries']['B']['total_kwh']==395
    assert expected['common_totals']=={'A':276,'B':395}
    assert len(expected['overlap'])==13


@pytest.mark.parametrize('kind',['plants','energy'])
def test_reversing_csv_order_does_not_change_answers(kind):
    rows = list(reversed(read_rows(kind)))
    result = extract_original(kind,calculation_cells(kind,rows))
    if kind=='plants':
        grade_plants(result,tolerance=5.1e-5)
    else:
        grade_energy(result)


@pytest.mark.parametrize('value',['',' ','  \t'])
def test_energy_blank_is_missing_not_zero(value):
    result = calculation_cells('energy',[erow(value)])
    assert len(result['missing_records'])==1
    assert not result['valid_records'] and not result['building_summaries']


@pytest.mark.parametrize('value',['0','0.0',' 0 ','-0.0'])
def test_energy_real_zero_is_retained(value):
    result = calculation_cells('energy',[erow(value)])
    assert result['building_summaries']['A']['available_readings']==1
    assert result['building_summaries']['A']['total_kwh']==0
    assert not result['missing_records']


@pytest.mark.parametrize('value',['NaN','nan','inf','-inf','Infinity','-1','garbage','1,234'])
def test_energy_nonfinite_negative_or_nonnumeric_rejected(value):
    result = calculation_cells('energy',[erow(value)],indices=(1,))
    assert len(result['rejected_records'])==1 and not result['valid_records']


@pytest.mark.parametrize('value,flagged',[('39.999',False),('40',True),('40.001',True)])
def test_energy_threshold_inclusive_and_flagged_rows_retained_in_sum(value,flagged):
    result = calculation_cells('energy',[erow(value)])
    assert bool(result['flagged_records'])==flagged
    assert result['building_summaries']['A']['total_kwh']==float(value)


@pytest.mark.parametrize('row',[erow(day='bad-date'),erow(day='2026-02-30'),erow(building='')])
def test_energy_bad_date_or_identifier_rejected(row):
    result = calculation_cells('energy',[row],indices=(1,))
    assert len(result['rejected_records'])==1 and not result['valid_records']


def test_energy_duplicate_key_does_not_double_sum():
    result = calculation_cells('energy',[erow('20'),erow('999')])
    assert result['building_summaries']['A']['total_kwh']==20
    assert len(result['rejected_records'])==1


def test_energy_no_common_date_means_empty_comparison():
    result = calculation_cells('energy',[erow(),erow(day='2026-01-02',building='B')])
    assert result['overlap_comparison']==[]


@pytest.mark.xfail(strict=True,reason='Original energy cell derives building universe from VALID rows; an all-missing B disappears, so A-only dates are incorrectly called overlap.')
def test_energy_building_with_all_missing_values_is_not_an_overlap():
    result = calculation_cells('energy',[erow(),erow('',building='B')])
    assert result['overlap_comparison']==[]


@pytest.mark.xfail(strict=True,reason='Original quality report enumerates untrimmed building IDs but counts trimmed IDs.')
def test_energy_whitespace_identifier_has_consistent_quality_counts():
    result = calculation_cells('energy',[erow(building=' A ')])
    assert result['validation_report']['per_building']=={'A':{'valid':1,'missing':0,'rejected':0}}


@pytest.mark.xfail(strict=True,reason='A short CSV row is represented by DictReader as None; original cell calls .strip() on None.')
def test_energy_truncated_csv_field_is_reported_not_crashed():
    result = calculation_cells('energy',[erow(None)])
    assert len(result['missing_records'])+len(result['rejected_records'])==1


@pytest.mark.parametrize('row',[prow(day='2026-02-30'),prow(height='abc'),prow(plant=''),prow(treatment=''),prow(height='')])
def test_plant_bad_required_value_is_reported(row):
    result = calculation_cells('plants',[row],indices=(2,))
    assert len(result['parse_errors'])==1 and not result['parsed_records']


def test_plant_insufficient_measurements_excluded():
    result = calculation_cells('plants',[prow()])
    assert result['plant_changes']==[]
    assert result['excluded_plants'][0]['plant_id']=='X'


def test_plant_endpoint_calculation_and_zero_height():
    result = calculation_cells('plants',[prow('10','2026-03-22'),prow('0','2026-03-01')])
    assert result['plant_changes'][0]['change_cm']==10
    assert result['plant_changes'][0]['first_height_cm']==0


@pytest.mark.parametrize('value',['NaN','inf','-inf'])
@pytest.mark.xfail(strict=True,reason='Original plant parser accepts non-finite float values and includes them as valid heights.')
def test_plant_nonfinite_height_is_rejected(value):
    result = calculation_cells('plants',[prow(value)],indices=(2,))
    assert result['parsed_records']==[] and len(result['parse_errors'])==1


@pytest.mark.xfail(strict=True,reason='Original plant cells do not reject duplicate plant/date observations; n_plants counts rows, not distinct plants.')
def test_plant_duplicate_key_not_counted_as_two_plants():
    result = calculation_cells('plants',[prow(),prow()])
    assert result['treatment_summary'][0]['n_plants']==1
    assert result['plant_changes']==[]


@pytest.mark.xfail(strict=True,reason='Original per-plant grouping silently chooses the first treatment when a plant changes treatment labels.')
def test_plant_inconsistent_treatment_is_not_silently_attributed():
    result = calculation_cells('plants',[prow(),prow('12','2026-03-22',treatment='fertilized')])
    assert result['plant_changes']==[]


@pytest.mark.xfail(strict=True,reason='Valid ISO inputs with mixed timezone awareness cannot be sorted by the original plant cell.')
def test_plant_mixed_timezone_requires_explicit_handling_not_crash():
    result = calculation_cells('plants',[prow(day='2026-03-01T00:00:00Z'),prow('12','2026-03-22')])
    assert len(result['plant_changes'])+len(result['excluded_plants'])==1


def test_plant_optional_columns_complete_case_behaviour_is_visible():
    result = calculation_cells('plants',[prow(leaf_count='')])
    assert result['parsed_records']==[]  # This original notebook uses complete-case exclusion.
    assert 'leaf_count is empty' in result['parse_errors'][0]['errors']


@pytest.mark.parametrize('mutate',[
    lambda r:r['summaries']['B'].update(total_kwh=315),  # silently deleting the high reading
    lambda r:r.update(missing_count=0),
    lambda r:r['common_totals'].update(A=301),  # mismatched date coverage
    lambda r:r['overlap'][0].update(difference_kwh=5),  # reversed sign
    lambda r:r['flagged'].clear(),
    lambda r:r['flagged'].append(dict(r['flagged'][0])),
    lambda r:r['summaries']['A'].update(mean_kwh=float('nan')),
])
def test_answer_checker_rejects_deliberately_wrong_energy_outputs(mutate):
    result = copy.deepcopy(energy_expected())
    mutate(result)
    with pytest.raises(AssertionError):
        grade_energy(result)


def test_knowledge_threshold_changes_expected_flags_not_totals():
    low,high = energy_expected(30),energy_expected(40)
    assert len(low['flagged'])==5 and len(high['flagged'])==1
    assert low['summaries']==high['summaries']
    with pytest.raises(AssertionError):
        grade_energy(low,40)
    grade_energy(low,30)


@pytest.mark.parametrize('value',[True,'301',float('nan'),float('inf')])
def test_numeric_grader_rejects_non_number_or_nonfinite(value):
    with pytest.raises(AssertionError):
        number(value,301,'test')
