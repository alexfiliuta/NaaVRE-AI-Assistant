"""Regression cases exposed by the live two-model energy-lab pilot."""
from naavre_ai_assistant.input_schema import schema_column_error

SCHEMA = [{'headers':['reading_date','building_id','electricity_kwh']}]

def test_named_aggregate_fields_in_constructed_records_are_not_source_columns():
    code = '''
summaries = []
for building, readings in grouped.items():
    count = len(readings)
    total = sum(readings)
    mean = total / count if count else None
    summaries.append({'building_id': building, 'count': count, 'total': total, 'mean': mean})
for row in summaries:
    print(row['count'], row['total'], row['mean'])
'''
    assert schema_column_error(code,SCHEMA) is None

def test_scalar_aggregate_names_can_be_used_in_a_later_cell():
    prior = '''
summaries = []
for building, values in grouped.items():
    available_readings = len(values)
    total_kwh = sum(values)
    summaries.append({'building_id':building,'available_readings':available_readings,'total_kwh':total_kwh})
'''
    assert schema_column_error("for row in summaries:\n    print(row['available_readings'],row['total_kwh'])",SCHEMA,prior) is None

def test_configuration_values_must_not_whitelist_source_typos():
    code = "settings={'building_id':'A','date':'reading_date'}\nprint(row['date'])"
    assert schema_column_error(code,SCHEMA)

def test_required_source_headers_still_reject_aggregate_fields():
    prior = "summary={'building_id':building,'count':len(readings)}"
    assert schema_column_error("required_columns=['reading_date','count']",SCHEMA,prior)

def test_misspelled_actual_source_lookup_is_rejected():
    code = "for row in data:\n    value=float(row['electricity_kw'])"
    assert schema_column_error(code,SCHEMA)

def test_converted_misspelled_source_is_not_a_safe_scalar_aggregate():
    code = "value=float(row['electricity_kw'])\nsummary={'building_id':building,'value':value}\nprint(row['value'])"
    assert 'electricity_kw' in schema_column_error(code, SCHEMA)

def test_configuration_name_is_not_a_calculated_field():
    code = "date='reading_date'\nsettings={'building_id':'A','date':date}\nprint(row['date'])"
    assert schema_column_error(code, SCHEMA)
