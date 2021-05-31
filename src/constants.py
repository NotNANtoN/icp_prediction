# constant values/ name strings

label_col = "target"

gender = ['ges_1', 'ges_2', 'ges_3', 'ges_4']

drop_variables_list = ["plz", "sernr", "altqx", "regbez", "lkrs", 'f1a_3', 'f1a_4',
                       'f1a_5', 'f1a_6', 'f1b_3', 'f1b_4', 'f1b_5', 'f1b_6', 'f2a_3', 'f2a_4',
                       'f2b_3', 'f2b_4'] + ['f1a_2', 'f1b_2', 'f2a_2', 'f2b_2']
                                                            # diagnosed

cols_for_outlier_removal = []

# contact or myself (tested or diagnosed) at any time (2 weeks ago or earlier) # after one-hot
compound_label_cols_incl_diagnosed = ['f1a_1', 'f1a_2', 'f1b_1', 'f1b_2', 'f2a_1', 'f2a_2',
                                      'f2b_1', 'f2b_2']

compound_label_cols_only_tested = ['f1a_1', 'f1b_1', 'f2a_1', 'f2b_1']



### Overview of different variable types ###

# variables with std below 0.02 - attention: naming here is after one-hot encoding
low_variance = ['f3_8', 'f41', 'f49', 'f145', 'f2a_1', 'f38_5', 'f43_4', 'f45_1', 'f45_3', 'f45_7', 'f46_3', 'f54_6', 'f137_3', 'f138_3', 'f139_3', 'f140_3', 'f141_3', 'f143_3']

# binary variables
contact_positive = ["f1a_1", "f1a_2", "f1a_3", "f1a_4", "f1a_5", "f1a_6"]
self_positive = ["f1b_1", "f1b_2", "f1b_3", "f1b_4", "f1b_5", "f1b_6"]
family_positive = ["f4a_1", "f4a_2", "f4a_3", "f4a_4", "f4b_1", "f4b_2", "f4b_3", "f4b_4"]

symptoms_bin = ["f3_1", "f3_2", "f3_3", "f3_4", "f3_5", "f3_6", "f3_7", "f3_8", "f3_9", "f3_10",
                "f3_11", "f3_12", "f3_13", "f3_16", "f3_17", "f3_18", "f3_19", "f3_20", "f145"]

pos_changes_bin = ['f8_1', 'f8_2', 'f8_3', 'f8_4', 'f8_5', 'f8_6', 'f8_7', 'f8_8', 'f8_9', 'f8_10', 'f8_11', 'f8_12', 'f8_13', 'f8_14', 'f8_15', 'f8_16', 'f8_17', 'f8_18', 'f8_19', 'f8_20', 'f8_21', 'f8_22']
canceled_event_bin = ["f17"]
reasons_less_work = ['f44_1', 'f44_2', 'f44_3', 'f44_4', 'f44_5', 'f44_6', 'f44_7', 'f44_8']


# ordinal - set top value to mean ("weiß nicht")
ordinal_questions = ["f5", "f6", "f7", "f9", "f10", 'f11', 'f12', 'f13', 'f14', 'f15', 'f16',
                     'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28',
                     'f29', 'f30', 'f31', 'f32', 'f37', 'f63', 'f64', 'f68', 'f72', 'f89', 'f95',
                     'f96', 'f98', 'f99', 'f100', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107',
                     'f108', 'f109', 'f110', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116',
                     'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125',
                     'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f146']

preconditions_when = ['f135a', 'f135b', 'f135c', 'f135d', 'f135e', 'f135f', 'f135g', 'f135h', 'f135i', 'f135j', 'f135k', 'f135l', 'f135m', 'f135n', 'f135o', 'f135p', 'f135q', 'f135r']

# interval - 99 == nan
interval_questions = ['nf33a', 'nf33b', 'nf34a', 'nf34b', 'nf35a', 'nf35b', 'nf36a', 'nf36b',
                      'nf42', 'nf76', 'nf77', "nf82", "nf83", "nf84", 'nf88', 'nf92', 'nf94',
                      'nt1', 'nt2']

non_categorical = ordinal_questions + preconditions_when + interval_questions

# -1 == nan
minus_nan = ["weight", "altq", "ges"]

# to be one-hot encoded

to_one_hot = ["f2a", "f2b", "f38", "f39", "f40", "f43", 'f44_1', 'f44_2', 'f44_3', 'f44_4',
              'f44_5', 'f44_6', 'f44_7', 'f44_8', "f45", "f46", "f47", 'f50', 'f51', 'f52', 'f53',
              'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f65_1', 'f65_2',
              'f65_3', 'f65_4', 'f65_5', 'f66', 'f67', 'f69', 'f70', 'f73', 'f74', 'f75', 'f85',
              'f86', 'f87', 'expe', 'f90', 'f91', 'f93', 'f97', 'f136', 'f137', 'f138', 'f139',
              'f140', 'f141', 'f142', 'f143', 'f144', 'ges', 'bl']
expect_change = ['f65_1', 'f65_2', 'f65_3', 'f65_4', 'f65_5', 'f66', 'f67']
reduced_income = ['f71_1', 'f71_2', 'f71_3', 'f71_4', 'f71_5', 'f71_6', 'f71_7', 'f71_8']


# open-ended

open_ended = ["f3_14", "f3_15"]  # as well as the ones with "open" in their names

# "Alter/ Schulkind/ Hilfe Hausaufgaben/ Besuch Kindertagesstätte Kind 1, 2, ... " - would ignore
# those
# at first
age_kids = ['f78_1', 'f78_2', 'f78_3', 'f78_4', 'f78_5', 'f78_6', 'f79_1', 'f79_2', 'f79_3',
            'f79_4', 'f79_5', 'f79_6', 'f80_1', 'f80_2', 'f80_3', 'f80_4', 'f80_5', 'f80_6', 'f81_1', 'f81_2', 'f81_3', 'f81_4', 'f81_5', 'f81_6']

not_always_applicable = ['f48_1', 'f48_2', 'f48_3', 'f48_4', 'f48_5','f48_6','f48_7','f48_8']
# a lot of missings there bc. they are conditional questions