import pandas as pd
import numpy as np
import statsmodels.api as sm

# Exam scores data from this page:
# http://www.bristol.ac.uk/cmm/learning/support/datasets/

# +
colspecs = [(0, 5), (6, 10), (11, 12), (13, 16), (17, 20)]
df = pd.read_fwf("../data/exam_scores/SCI.DAT", colspecs=colspecs, header=None)
df.columns = ["schoolid", "subjectid", "gender", "score1", "score2"]
df["female"] = 1*(df.gender == 1)
df = df.dropna()
# -

# +
# A school-clustered model for exam score 1 with no correlation.
model1 = sm.GEE.from_formula("score1 ~ female", groups="schoolid", data=df)
rslt1 = model1.fit()
print(rslt1.summary())
# -

# +
# A school-clustered model for exam score 1 with exchangeable correlations.
model2 = sm.GEE.from_formula("score1 ~ female", groups="schoolid",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt2 = model2.fit()
print(rslt2.summary())
# -

# +
# A subject-clustered model for exam score 1 with exchangeable correlations.
model3 = sm.GEE.from_formula("score1 ~ female", groups="subjectid",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt3 = model3.fit()
print(rslt3.summary())
# -

# +
# Prepare to do a joint analysis of the two scores.
dx = pd.melt(df, id_vars=["subjectid", "schoolid", "female"],
             value_vars=["score1", "score2"], var_name="test",
             value_name="score")
# -

# +
# A nested model for subjects within schools, having two scores per subject.
model4 = sm.GEE.from_formula("score ~ female + test", groups="schoolid", dep_data="0 + subjectid",
                             cov_struct=sm.cov_struct.Nested(), data=dx)
rslt4 = model4.fit()
print(rslt4.summary())
# -