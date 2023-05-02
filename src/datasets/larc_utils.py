import itertools

GRADE_TASK = "grade"
RETENTION_TASK = "retention"

MISSING_VALUE = "MISSING"  # use this for categorical data with missing value

COURSE_TYPES = ["LEC", "DIS", "SEM", "LAB", "others", "MISSING"]

# COURSE_COMBINED_TYPES contains the following:
# ['LEC-SEM', 'LEC-others', 'DIS-LEC', 'DIS-SEM', 'DIS-LAB', 'DIS-others',
# 'SEM-others', 'LAB-LEC', 'LAB-SEM', 'LAB-others']
COURSE_COMBINED_TYPES = [
    "-".join(c) for c in
    itertools.product(COURSE_TYPES, COURSE_TYPES)
    if c[0] < c[1]]

CIP2_CATEGORIES = ["{:02}".format(i) for i in range(1, 62)] + [MISSING_VALUE, ]
