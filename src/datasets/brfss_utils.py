"""

Utilities for working with BRFSS dataset.

Accessed via https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system.
Raw Data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
Data Dictionary: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""
# Brief feature descriptions below; for the full question/description
# see the data dictionary linked above.
BRFSS_FEATURES = [
    ################ Target ################
    "DIABETE3",  # (Ever told) you have diabetes

    # ################ Demographics/sensitive attributes. ################
    # Also see "INCOME2", "MARITAL", "EDUCA" features below.
    "_STATE",
    # Was there a time in the past 12 months when you needed to see a doctor 
    # but could not because of cost?
    "MEDCOST",
    # Respondents aged 18-64 who have any form of health care coverage
    "_HCVU651",
    # Preferred race category; note that ==1 is equivalent to 
    # "White non-Hispanic race group" variable _RACEG21
    "_PRACE1",
    # Indicate sex of respondent.
    "SEX",

    # Below are a set of indicators for known risk factors for diabetes.

    ################ General health ################
    # for how many days during the past 30 days was your 
    # physical health not good?
    "PHYSHLTH",
    ################ High blood pressure ################
    # Adults who have been told they have high blood pressure by a 
    # doctor, nurse, or other health professional
    "_RFHYPE5",
    ################ High cholesterol ################
    # Cholesterol check within past five years
    "_CHOLCHK",
    # Have you EVER been told by a doctor, nurse or other health 
    # professional that your blood cholesterol is high?
    "TOLDHI2",
    ################ BMI/Obesity ################
    # Calculated Body Mass Index (BMI)
    "_BMI5",
    # Four-categories of Body Mass Index (BMI)
    "_BMI5CAT",
    ################ Smoking ################
    # Have you smoked at least 100 cigarettes in your entire life?
    "SMOKE100",
    # Do you now smoke cigarettes every day, some days, or not at all?
    "SMOKDAY2",
    ################ Other chronic health conditions ################
    # (Ever told) you had a stroke.
    "CVDSTRK3",
    # ever reported having coronary heart disease (CHD) 
    # or myocardial infarction (MI)
    "_MICHD",
    ################ Diet ################
    # Consume Fruit 1 or more times per day
    "_FRTLT1",
    # Consume Vegetables 1 or more times per day
    "_VEGLT1",
    ################ Alcohol Consumption ################
    # Calculated total number of alcoholic beverages consumed per week
    "_DRNKWEK",
    # Binge drinkers (males having five or more drinks on one occasion, 
    # females having four or more drinks on one occasion)
    "_RFBING5",
    ################ Exercise ################
    # Adults who reported doing physical activity or exercise 
    # during the past 30 days other than their regular job
    "_TOTINDA",
    # Minutes of total Physical Activity per week
    "PA1MIN_",
    ################ Household income ################
    # annual household income from all sources
    "INCOME2",
    ################ Marital status ################
    "MARITAL",
    ################ Time since last checkup
    # About how long has it been since you last visited a 
    # doctor for a routine checkup?
    "CHECKUP1",
    ################ Education ################
    # highest grade or year of school completed
    "EDUCA",
    ################ Health care coverage ################
    # Respondents aged 18-64 who have any form of health care coverage
    "_HCVU651",
    ################ Mental health ################
    # for how many days during the past 30 
    # days was your mental health not good?
    "MENTHLTH",
]
