
import pandas as pd
import numpy as np
import os
import tqdm

# this scripts loads the real data and generates fake data under the same schema
fn = os.path.join('C:/users/moehring/git/SurfacingNormsToIncreaseVaccineAcceptance/data/covid_survey_responses_numeric.txt.gz')
df = pd.read_table(fn, sep='\t', low_memory=False)

df = df.loc[df.display_order.apply(lambda v: 'surveyresponseinformation' in str(v)) & (df.progress == 100) & (df.survey_type == 'waves') & (df.wave == 15)].iloc[0:50000]
df['id'] = range(len(df))
df['eligible_for_information'] = True
for col in tqdm.tqdm(df.columns):
    # randomly sample data from columns
    df[col] = np.random.choice(df[col].unique(), size=len(df), replace=True)
df['eligible_for_information'] = True
df['wave'] = 1
df['survey_response_information_version'] = 1
df['survey_information_value'] = df.survey_information_level.apply(lambda v: 50 if v == 'low' else 70)
df.to_csv(os.path.join(os.path.dirname(fn), 'random_demo_data.txt.gz'), index=False, sep='\t')
