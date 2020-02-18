import pandas as pd
import numpy as np
import statsmodels.api as sm

# # GEE analysis of test score data

# Generalized estimating equations (GEE) can be used to analyze
# multilevel data that arises in educational research, for example
# when students taking a test are grouped into classrooms.

# The examination data analyzed here are obtained from this page:
# http://www.bristol.ac.uk/cmm/learning/support/datasets/

# The data are in fixed-width format, we can load it as follows:

# +
colspecs = [(0, 5), (6, 10), (11, 12), (13, 16), (17, 20)]
df = pd.read_fwf("../data/exam_scores/SCI.DAT", colspecs=colspecs, header=None)
df.columns = ["schoolid", "subjectid", "gender", "score1", "score2"]
df["female"] = 1*(df.gender == 1)
df = df.dropna()
# -

# Here is a basic model looking at the scores on exam 1 by gender, using
# the default independence working correlation structure.

# +
# A school-clustered model for exam score 1 with no correlation.
model1 = sm.GEE.from_formula("score1 ~ female", groups="schoolid", data=df)
rslt1 = model1.fit()
print(rslt1.summary())
# -

# Here is the same mean structure model, now specifying that the students
# are exchangeably correlated within classrooms.

# +
# A school-clustered model for exam score 1 with exchangeable correlations.
model2 = sm.GEE.from_formula("score1 ~ female", groups="schoolid",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt2 = model2.fit()
print(rslt2.summary())
print(model2.cov_struct.summary())
# -

# Next we will pivot the exam scores so that each subject has two observations
# on a single "test" variable (one observation for the first test and one for
# the second test).  This is a form of repeated measures, but since the tests
# are different, we also include a covariate indicating which test is being
# recorded.  We now have two levels of repeated structure: two test scores per
# student, and multiple students per classroom.  We can use a nested correlation
# structure to estimate the variance contributions from the two levels.

# +
# Prepare to do a joint analysis of the two scores.
dx = pd.melt(df, id_vars=["subjectid", "schoolid", "female"],
             value_vars=["score1", "score2"], var_name="test",
             value_name="score")
# -

# +
# A nested model for subjects within schools, having two scores per subject.
model3 = sm.GEE.from_formula("score ~ female + test", groups="schoolid", dep_data="0 + subjectid",
                             cov_struct=sm.cov_struct.Nested(), data=dx)
rslt3 = model3.fit()
print(rslt3.summary())
print(model3.cov_struct.summary())
# -