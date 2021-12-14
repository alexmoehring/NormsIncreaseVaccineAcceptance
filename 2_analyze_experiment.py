
import pandas as pd
import scipy.stats
import datetime as dt
import os
import json
import numpy as np
import patsy
import statsmodels.formula.api as smf
import matplotlib
from matplotlib import pyplot as plt
from stargazer.stargazer import Stargazer
import pickle
import logging
import tqdm


############################################
# Setting initial global parameters and paths #
############################################

# Formatting parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['patch.antialiased'] = True
matplotlib.rcParams['patch.linewidth'] = 0.5

color_map = {
    'Broad': '#093469',
    'Narrow': '#92c2fc'
}
other_colors = ['#C98362', '#D6B05D']

# File paths
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'norms_experiment')
if not os.path.exists(out_path):
    os.makedirs(out_path)
raw_fig_path = os.path.join(out_path, 'figs', '{0}'.format(dt.datetime.utcnow().strftime('%Y%m%d')))
fig_paths = {
    'weight_full': os.path.join(raw_fig_path, 'weight_full'),
    'unweighted': os.path.join(raw_fig_path, 'unweighted')
}

# building directory
for fp_raw in fig_paths:
    tmp_fp = os.path.join(raw_fig_path, fp_raw)
    if not os.path.exists(tmp_fp):
        os.makedirs(tmp_fp)
    if not os.path.exists(os.path.join(tmp_fp, 'robustness')):
        os.makedirs(os.path.join(tmp_fp, 'robustness'))

# set up logger
log = logging.getLogger('covid_survey')
log.setLevel(logging.DEBUG)

# create console handler and set level info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
log.addHandler(handler)

# maps to be used later on
block2col = {
    'demographics': ['age', 'gender', 'country', 'education', 'own_health', 'density'],
    'vaccineandhealthcare': ['vaccine_accept', 'knowledge_existing_treatments'],
    'covid-19informationexposure': ['info_exposure_past_week', 'info_exposure_more_less_wanted'],
    'knowledgeandpositivecases': ['know_positive_case', 'knowledge_existing_treatments'],
    'information': ['news_medium_*', 'news_sources_*'],
    'preventionbehaviorsinpractice': ['prevention_mask', 'prevention_distancing', 'prevention_hand_washing'],
    'ben': ['effect_mask', 'measures_taken_*', 'effect_hand_washing', 'country_management', 'community_management', 'community_action_importance',
            'community_action_norms'],
    'distancingfamiliarity,importance&norms': ['distancing_importance', 'distancing_norms_maintain_a_distance_of_at_least_1_meter_from_others',
                                               'distancing_norms_wear_a_face_mask_or_covering', 'norms_vaccine'],
    'futureactions': ['future_vaccine', 'future_masks', 'future_dist'],
    'riskperceptionsandlocusofcontrol': ['risk_community', 'risk_infection', 'control_infection', 'infection_severity'],
    'work': ['employed_2020', 'work_changes', 'work_type', 'work_industry'],
    'intentionstovisit': ['locations_would_attend_*'],
    'surveyresponseinformation': []
}
col2block = {}
for tmp_b in block2col:
    for tmp_c in block2col[tmp_b]:
        col2block[tmp_c] = tmp_b
cleancol2block = {'country_cov': 'demographics'}

group_map = {
    'high': 'Broad',
    'low': 'Narrow'
}

reverse_label_map = {
    "vaccine_accept": {
        "Yes": 2,
        "Don't know": 1,
        "No": 0
    },
    "future_masks": {
        "Always": 4,
        "Almost always": 3,
        "When convenient": 2,
        "Rarely": 1,
        "Never": 0
    },
    "future_dist": {
        "Always": 4,
        "Almost always": 3,
        "When convenient": 2,
        "Rarely": 1,
        "Never": 0
    },
    "future_vaccine": {
        "Yes, definitely": 4,
        "Probably": 3,
        "Unsure": 2,
        "Probably not": 1,
        "No, definitely not": 0
    }
}
label_map = {}
for q_name in reverse_label_map:
    label_map[q_name] = {}
    for answer_text in reverse_label_map[q_name]:
        num_answer = reverse_label_map[q_name][answer_text]
        assert num_answer not in label_map[q_name]
        label_map[q_name][num_answer] = answer_text


# screens per block
block2screens = {
    'demographics': 7,
    'vaccineandhealthcare': 2,
    'covid-19informationexposure': 1,
    'knowledgeandpositivecases': 2,
    'information': 3,
    'preventionbehaviorsinpractice': 2,
    'ben': 5,
    'distancingfamiliarity,importance&norms': 2,
    'futureactions': 2,
    'riskperceptionsandlocusofcontrol': 2,
    'work': 1,
    'intentionstovisit': 1,
    'surveyresponseinformation': 1,
    'basicknowledge': 3
}
constants = {}

###########################
# preprocessing functions #
###########################


def weighted_mean(series, weights):
    # remove elements w/ null values or weights
    ixs = ~(pd.isna(series) | pd.isna(weights) | (weights <= 0))
    to_ret = np.average(series[ixs], weights=weights[ixs])
    assert not pd.isna(to_ret)
    return to_ret


def weighted_std(series, weights):
    # remove elements w/ null values or weights
    ixs = ~(pd.isna(series) | pd.isna(weights) | (weights <= 0))
    series = series[ixs]
    weights = weights[ixs]
    mean = weighted_mean(series, weights)
    squared_diff = (series - mean)**2
    var_est = weighted_mean(squared_diff, weights) * len(squared_diff) / (len(squared_diff) - 1)  # DF correction
    to_ret = var_est**0.5
    assert not pd.isna(to_ret)
    return to_ret


def process_data(df):

    # add block positions
    for b in block2col:
        df['{0}_pos'.format(b)] = df.display_order.apply(lambda v: v['block_order'][b] if b in v['block_order'] else np.nan)
        df['{0}_pos_parent'.format(b)] = df.display_order.apply(lambda v: v['parent_block_order'][b] if b in v['parent_block_order'] else np.nan)

    for b in tqdm.tqdm(block2col):
        # get pre/post
        locs = {
            '_pre': df[b + '_pos'] < df['surveyresponseinformation_pos'],
            '_pre_both': (df[b + '_pos'] < df['surveyresponseinformation_pos']) & (df[b + '_pos'] < df['futureactions_pos']),
        }
        for tmp_q in block2col[b]:
            if '*' in tmp_q:
                qs = [el for el in df.columns if tmp_q.replace('*', '') in el]
            else:
                qs = [tmp_q]
            for q in qs:
                for suffix in locs:
                    if pd.Series(locs[suffix]).sum() == 0:
                        continue
                    clean_q = q
                    if q == 'distancing_norms_maintain_a_distance_of_at_least_1_meter_from_others':
                        clean_q = 'norms_dist'
                    elif q == 'distancing_norms_wear_a_face_mask_or_covering':
                        clean_q = 'norms_masks'
                    cleancol2block[clean_q] = b
                    if clean_q not in df.columns:
                        df[clean_q] = df[q]
                    df.loc[locs[suffix], clean_q + suffix] = df.loc[locs[suffix], q]
                    if df[q].dtype != 'object':
                        if q == 'vaccine_accept':
                            with open('maps.json', 'r') as f:
                                dont_know_val = json.load(f)['numeric_map']['vaccine_accept']["Don't know"]
                            df[clean_q + suffix + '_imputed'] = df[clean_q + suffix].fillna(dont_know_val)  # pre-registered this
                            df[clean_q + '_imputed'] = df[clean_q].fillna(dont_know_val)
                        else:
                            df[clean_q + suffix + '_imputed'] = df[clean_q + suffix].fillna(weighted_mean(df[clean_q + suffix], weights=df.weight_full_survey))
                            df[clean_q + '_imputed'] = df[clean_q].fillna(weighted_mean(df[clean_q], weights=df.weight_full_survey))
    return df


def add_baseline_belief_partition(df):
    # add baseline belief partition
    # first get high and low values four country/period
    behaviors = [el for el in df.behavior.unique() if not pd.isna(el) and el != 'control']
    countries = [el for el in df.country.unique() if not pd.isna(el)]
    nbins = 5

    for behavior in behaviors:
        for col in ['baseline_belief_partition_{0}_pre'.format(behavior),
                    'baseline_belief_partition_{0}_pre_both'.format(behavior)]:
            if col.endswith('_both'):
                norms_col = 'norms_{0}_pre_both'.format(behavior)
                linear_col = 'baseline_diff_linear_{0}_' + behavior + '_pre_both'
            else:
                norms_col = 'norms_{0}_pre'.format(behavior)
                linear_col = 'baseline_diff_linear_{0}_' + behavior + '_pre'
            granular_col = linear_col.replace('linear', 'granular')
            df[col] = np.nan
            df['high_value_{0}'.format(behavior)] = np.nan
            df['low_value_{0}'.format(behavior)] = np.nan
            for country in countries:
                for period in df.period.unique():
                    tmp = df.loc[(df.behavior == behavior) & (df.country == country) & (df.period == period)]
                    if len(tmp) == 0:
                        continue
                    high_val = tmp.loc[tmp.level == 'high'].value.unique()
                    low_val = tmp.loc[tmp.level == 'low'].value.unique()
                    assert len(high_val) == 1, 'Behavior: {0}, Country: {1}, period: {2}'.format(behavior, country, period)
                    assert len(low_val) == 1, 'Behavior: {0}, Country: {1}, period: {2}'.format(behavior, country, period)
                    high_val = high_val[0]
                    low_val = low_val[0]
                    assert high_val > low_val

                    # Add partition
                    ixs = (df.country == country) & (df.period == period)
                    assert np.all(pd.isna(df.loc[ixs, col]))
                    df.loc[ixs & (df[norms_col] < low_val), col] = 'Under'
                    df.loc[ixs & (df[norms_col] >= low_val) & (df[norms_col] < high_val), col] = 'Between'
                    df.loc[ixs & (df[norms_col] >= high_val), col] = 'Above'

                    # add high and low stimuli
                    assert (~pd.isna(df.loc[ixs, 'high_value_{0}'.format(behavior)])).sum() == 0
                    assert (~pd.isna(df.loc[ixs, 'low_value_{0}'.format(behavior)])).sum() == 0
                    df.loc[ixs, 'high_value_{0}'.format(behavior)] = high_val
                    df.loc[ixs, 'low_value_{0}'.format(behavior)] = low_val

            assert granular_col != linear_col
            for level in ['high', 'low']:
                level_linear_col = linear_col.format(level)
                level_granular_col = granular_col.format(level)
                assert level_linear_col not in df.columns and level_granular_col not in df.columns
                val_col = '{0}_value_{1}'.format(level, behavior)

                df[level_linear_col] = df[norms_col] - df[val_col]  # positive == above treatment
                pos_cutoffs = df.loc[df[level_linear_col] >= 0, [level_linear_col]].quantile([el / nbins for el in range(1, nbins + 1)]).sort_index()
                neg_cutoffs = df.loc[df[level_linear_col] < 0, [level_linear_col]].quantile([el / nbins for el in range(1, nbins + 1)]).sort_index()
                neg_cutoffs.index = ['neg_{0}'.format(el) for el in neg_cutoffs.index]
                pos_cutoffs.index = ['pos_{0}'.format(el) for el in pos_cutoffs.index]
                cutoffs = pd.concat([neg_cutoffs, pos_cutoffs], axis=0)
                val2group = {}
                for el in df[level_linear_col].unique():
                    for ix in cutoffs.index:
                        if el <= cutoffs.loc[ix, level_linear_col]:
                            val2group[el] = cutoffs.loc[ix, level_linear_col]
                            break
                df[level_granular_col] = df[level_linear_col].apply(lambda v: val2group[v] if not pd.isna(v) else np.nan)
    return df


def normalize_weights(df):
    # for each wave, normalize weights so they sum to the number of responses in each country
    weight_cols = [el for el in df.columns if 'weight' in el]
    for wave in df.wave.unique():
        for country in df.country.unique():
            if pd.isna(country) or pd.isna(wave):
                continue
            for c in weight_cols:
                ixs = pd.Series((df.country == country) & (df.wave == wave) & ~pd.isna(df[c]))
                num_responses = ixs.sum()
                df.loc[ixs, c] = df.loc[ixs, c] / df.loc[ixs, c].sum() * num_responses
    return df


#################
# baseline figs #
#################

def plot_treatments(df):
    # plot information treatment values by country and version
    agg = df.groupby(['country_pre', 'behavior', 'level', 'survey_response_information_version'],
                     as_index=False).value.mean()

    for b in [el for el in agg.behavior.unique()]:
        tmp = agg.loc[agg.behavior == b].copy()
        tmp['group'] = tmp.level
        tmp.group = tmp.group.apply(lambda v: group_map[v])

        fn = os.path.join(raw_fig_path, 'treatment_plot_by_country_{0}.pdf'.format(b))
        tmp['Estimate'] = tmp.value
        tmp['SE'] = 0
        tmp.index = tmp.country_pre
        country_order = list(tmp.sort_values('Estimate').country_pre.unique())

        # now add actual norm to highlight mismatch
        for country in tmp.index:
            col = 'norms_{0}_pre'.format(b)
            country_df = df.loc[(df.country_pre == country) & ~pd.isna(df[col]) & ~pd.isna(df.weight_demo)]
            for info_version in df.survey_response_information_version.unique():
                # calculate average belief about norm
                tdf = country_df.loc[(country_df.survey_response_information_version == info_version)]
                col = 'norms_{0}_pre'.format(b)
                avg_norm = weighted_mean(tdf[col], weights=tdf['weight_full_survey'])
                to_add = pd.DataFrame(0, index=[country], columns=tmp.columns)
                to_add['country_pre'] = country
                to_add['Estimate'] = avg_norm
                to_add['SE'] = 0
                to_add['group'] = 'Country Belief'
                tmp = pd.concat([tmp, to_add])

        tmp = tmp.loc[[el for el in country_order]]
        tmp_color_map = color_map
        tmp_color_map['Country Belief'] = other_colors[0]

        # title = b.capitalize()
        ChartTools.plot_coefs(coefs=tmp, fn=fn, title='', pickle_fig=False, offset_amt=0, color_map=tmp_color_map, line_loc=None, xlim=(0, 100))


###################
# balance checks  #
###################


def test_randomization(d, treatment_indicator, sample_name):
    """
    Tests randomization that share of treatment equals share of controls for all preventative behaviors
    """
    log.info('\nRandomization Checks: {0} & {1}'.format(treatment_indicator, sample_name))
    log.info('Num samples: {0}'.format(len(d)))
    tmp = d.loc[~pd.isna(d[treatment_indicator])]
    num_controls = (tmp[treatment_indicator] == 'control').sum()
    num_treated = (tmp[treatment_indicator] != 'control').sum()
    num_null = pd.isna(tmp[treatment_indicator]).sum()
    assert num_controls + num_treated + num_null == len(tmp), 'Controls: {0}, Treated: {1}, Null: {2}'.format(num_controls, num_treated, num_null)
    log.info(tmp[treatment_indicator].value_counts())
    shares = tmp[treatment_indicator].value_counts() / tmp[treatment_indicator].value_counts().sum()
    p = scipy.stats.binom_test(x=num_controls, n=num_controls + num_treated, p=0.5)
    log.info(shares)
    log.info(p)
    return {
        'p': p,
        'treated': 1 - num_controls / (num_controls + num_treated),
        'control': num_controls / (num_controls + num_treated)
    }


def test_randomization_within(d, treatment_indicator, sample_name, wave2proportion=None):
    """
    Tests randomization within treated individuals. Tests that the share of each preventative
    behavior are as expected. This has to be careful because the treatment probabilities changed
    across the waves (see manuscript for details).
    """
    log.info('\nRandomization Checks: {0} & {1}'.format(treatment_indicator, sample_name))
    log.info('Num samples: {0}'.format(len(d)))

    # just manually center things
    tmp = d.loc[~pd.isna(d[treatment_indicator])].copy()
    tmp['tmp_treatment'] = pd.Series(tmp[treatment_indicator] != 'control').astype(int)

    formula = 'tmp_treatment ~ 0 + period'
    mod = RegressionTools.lm_robust(df=tmp, formula=formula, se_type='HC2')
    tests = []
    for el in mod.params.index:
        # get expected proportion
        p = 0.5
        if not isinstance(wave2proportion, type(None)):
            wave_num = int(el.split('_')[0].split('wave')[-1])
            p = wave2proportion[wave_num]
        tests.append('{0}={1}'.format(el, p))
    test_results = mod.wald_test(', '.join(tests))
    log.info(test_results)
    return {
        'test': test_results,
        'p': test_results.pvalue,
        'treated': tmp.tmp_treatment.mean(),
        'control': 1 - tmp.tmp_treatment.mean()
    }


def test_balance(d, sample_name, weights, block_locs=None):
    log.info('\nBalance Checks: {0}'.format(sample_name))
    balance_vars = [el for el in d.columns if '_pre_both' in el and el + '_imputed' in d.columns and 'future_' not in el and el != 'norms_pre']
    balance_table = pd.DataFrame()

    if isinstance(block_locs, type(None)):
        block_locs = list(range(1000))

    for v in balance_vars:
        tmp = d.loc[~pd.isna(d[v])]

        clean_v = v.replace('_pre', '').replace('_imputed', '').replace('_both', '')
        block = cleancol2block[clean_v]
        tmp = tmp.loc[tmp.display_order.apply(lambda do: False if block not in do['block_order'] else do['block_order'][block] in block_locs)]
        if len(tmp) == 0:
            continue

        # can calculate balance using t tests
        control_obs = tmp.loc[(tmp['intent_to_treat'] == 'control'), v]
        treated_obs = tmp.loc[(tmp['intent_to_treat'] == 'treated'), v]

        # but better to use regression so we can adjust for changes in sampling frequencies
        mod = RegressionTools.lm_lin_manual(df=tmp, covariates=['C(period)'], tv='intent_to_treat', ov=v,
                                            weights=tmp[weights], se_type='HC2')

        balance_table.loc[clean_v, 'p-val'] = mod.pvalues['intent_to_treat[T.treated]']
        balance_table.loc[clean_v, 'Control'] = control_obs.mean()
        balance_table.loc[clean_v, 'C STD'] = control_obs.std() / np.sqrt(len(control_obs))
        balance_table.loc[clean_v, 'Treated'] = treated_obs.mean()
        balance_table.loc[clean_v, 'T STD'] = treated_obs.std() / np.sqrt(len(treated_obs))
        balance_table.loc[clean_v, 'C Count'] = len(control_obs)
        balance_table.loc[clean_v, 'T Count'] = len(treated_obs)

    log.info(balance_table)

    balance_table.index = [el.replace('_pre', '').replace('_both', '').replace('_', ' ') for el in balance_table.index]
    directory = os.path.join(raw_fig_path, 'randomization_checks')
    # plot p values
    to_plot = balance_table['p-val'].sort_values()
    to_plot = to_plot.to_frame()
    to_plot['x'] = range(len(to_plot))
    fig, ax = plt.subplots()
    to_plot.plot(ax=ax, kind='scatter', x='x', y='p-val')
    ChartTools.save_show_plot(fig=fig, fn=os.path.join(directory, 'balance_p_values_{0}.pdf'.format(sample_name)), show_graph=False, pickle_fig=False)

    # save table
    to_save = balance_table[[el for el in balance_table.columns if 'Count' not in el]].copy()
    for el in [col for col in to_save.columns if 'STD' in col]:
        to_save[el] = to_save[el].apply(lambda t: '({0:.3f})'.format(t))
    to_save = to_save.rename(columns={'C STD': '', 'T STD': ''})
    to_save.to_csv(os.path.join(raw_fig_path, 'randomization_checks', 'balance_table_mean_{0}.csv'.format(sample_name)))
    to_save = to_save.loc[[el for el in to_save.index if not el.startswith('news') and not el.startswith('measures') and not el.startswith('locations')]]
    log.info(to_save)
    to_save.to_latex(
        os.path.join(raw_fig_path, 'randomization_checks', 'balance_table_mean_{0}.tex'.format(sample_name)),
        float_format='{:0.3f}'.format
    )


def test_balance_within_treatments(d, sample_name, weights):
    log.info('\nBalance Checks W/n Treatments: {0}'.format(sample_name))
    balance_vars = [el for el in d.columns if '_pre' in el and el + '_imputed' in d.columns and 'future_' not in el and el != 'norms_pre' and '_pre_both' not in el]
    d = d.loc[(d.intent_to_treat == 'treated') & ~pd.isna(d.behavior)].copy()
    mean_balance_table = pd.DataFrame()

    for v in balance_vars:
        tmp = d.loc[~pd.isna(d[v])].copy()

        masks_obs = tmp.loc[tmp.behavior == 'masks', v]
        vaccine_obs = tmp.loc[tmp.behavior == 'vaccine', v]
        dist_obs = tmp.loc[tmp.behavior == 'dist', v]

        # want vaccine to be the control
        tmp.loc[tmp.behavior == 'vaccine', 'behavior'] = 'a_behavior'

        mod = RegressionTools.lm_lin_manual(df=tmp, covariates=['C(period)'], tv='behavior', ov=v,
                                            weights=tmp[weights], se_type='HC2', sm_fit_collinear=True)
        mean_balance_table.loc[v, 'VD p-val'] = mod.pvalues['behavior[T.dist]']
        mean_balance_table.loc[v, 'VM p-val'] = mod.pvalues['behavior[T.masks]']
        mean_balance_table.loc[v, 'Vaccine'] = vaccine_obs.mean()
        mean_balance_table.loc[v, 'V STD'] = vaccine_obs.std() / np.sqrt(len(vaccine_obs))
        mean_balance_table.loc[v, 'Masks'] = masks_obs.mean()
        mean_balance_table.loc[v, 'M STD'] = masks_obs.std() / np.sqrt(len(masks_obs))
        mean_balance_table.loc[v, 'Dist'] = dist_obs.mean()
        mean_balance_table.loc[v, 'D STD'] = dist_obs.std() / np.sqrt(len(dist_obs))

    log.info(mean_balance_table)
    # plot p values
    for b in ['VD', 'VM']:
        to_plot = mean_balance_table['{0} p-val'.format(b)].sort_values()
        to_plot = to_plot.to_frame()
        to_plot['x'] = range(len(to_plot))
        fig, ax = plt.subplots()
        to_plot.plot(ax=ax, kind='scatter', x='x', y='{0} p-val'.format(b))
        ChartTools.save_show_plot(fig=fig, fn=os.path.join(raw_fig_path, 'randomization_checks', 'balance_p_values_within_{0}_{1}.pdf'.format(b, sample_name)), show_graph=False, pickle_fig=False)

    # save table
    mean_balance_table.index = [el.replace('_pre', '').replace('_both', '').replace('_', ' ') for el in mean_balance_table.index]
    to_save = mean_balance_table[[el for el in mean_balance_table.columns if 'Count' not in el]].copy()
    for el in [col for col in to_save.columns if 'STD' in col]:
        to_save[el] = to_save[el].apply(lambda t: '({0:.3f})'.format(t))
    to_save = to_save.drop(columns=['V STD', 'M STD', 'D STD'])
    to_save.to_csv(os.path.join(raw_fig_path, 'randomization_checks', 'balance_table_mean_within_{0}.csv'.format(sample_name)))
    to_save = to_save.loc[[el for el in to_save.index if not el.startswith('news') and not el.startswith('measures') and not el.startswith('locations')]]
    to_save.to_latex(
        os.path.join(raw_fig_path, 'randomization_checks', 'balance_table_mean_within_{0}.tex'.format(sample_name)),
        float_format='{:0.3f}'.format
    )

########################
# hte helper functions #
########################


def clean_covariate(s):
    """
    Helper function that cleans covariate strings
    """
    return s.replace('c(', '').replace(')', '').replace(' ', '').replace('factor(', '').replace('C(', '')


def f_test_restrictions_zero(cs, vars_to_test):
    r_mat = pd.DataFrame(0, index=vars_to_test, columns=cs.index)
    r_vec = np.zeros(len(r_mat))
    for el in vars_to_test:
        r_mat.loc[el, el] = 1
    return r_mat, r_vec


def reduced_form_analysis_single(d, name, ov, tv, covariates, weight_col):
    assert pd.isna(d[ov]).sum() == 0
    assert pd.isna(d[tv]).sum() == 0

    # loop through covariates and filter instances where there is only treated/control with a given variable
    # should only happen w/ futureaction_pos
    good_covariates = [el for el in covariates if len(d[clean_covariate(el)].unique()) > 1]
    for el in good_covariates:
        if 'C(' not in el:
            continue
        clean_c = clean_covariate(el)
        for v in d[clean_c].unique():
            if len(d.loc[d[clean_c] == v, tv].value_counts()) < len(d[tv].unique()):
                log.info('Dropping {0}: {1}'.format(el, v))
                d = d.loc[d[clean_c] != v].copy()

    # if len(d.outcome_type.unique()) > 1:
    #     good_covariates += ['outcome_type']
    mod = RegressionTools.lm_lin_manual(df=d, covariates=good_covariates, tv=tv, ov=ov,
                                        weights=d[weight_col], se_type='HC2', sm_fit_collinear=True)
    coefs = pd.DataFrame(index=mod.params.index, columns=['Estimate', 'SE'])
    coefs['Estimate'] = mod.params
    coefs['SE'] = mod.bse
    coefs['term'] = coefs.index
    terms2keep = [el for el in coefs.term if tv in el and ':' not in el]
    coefs = coefs.loc[terms2keep]
    coefs.index = [name] * len(coefs)
    coefs['group'] = [el.split('[T.')[1].replace(']', '') for el in coefs.term]
    log.info(coefs)
    assert pd.isna(coefs['Estimate']).sum() == 0
    assert pd.isna(coefs['SE']).sum() == 0
    return coefs, mod, d


def save_tables(fp, sg_models, lines, covariate_order, dependent_variable_name=None):
    # now save table
    sg = Stargazer(sg_models)
    sg.show_degrees_of_freedom(False)
    for l in lines:
        sg.add_line(label=l, values=lines[l])
    # sg.add_line(label='Cond No', values=[el.condition_number for el in sg_models])

    if not isinstance(dependent_variable_name, type(None)):
        dependent_variable_name = [str(el) for el in dependent_variable_name]
        sg.custom_columns(labels=dependent_variable_name, separators=[1 for _ in dependent_variable_name])
        sg.show_model_nums = False

    # save full table in html
    html = sg.render_html()
    assert '.tex' in fp
    tmp_fp = fp.replace('.tex', '_fulltable.html')
    with open(tmp_fp, 'w') as f:
        f.writelines(html)
    sg.show_header = False

    # filter to only show covariates of interest
    if not isinstance(covariate_order, type(None)):
        sg.covariate_order(cov_names=covariate_order)

    # rename rows removing underscores
    cov_map = {}
    for cov_name in sg.cov_names:
        if 'tmp_treatment[T.low]' in cov_name:
            clean_cov_name = cov_name.replace('tmp_treatment[T.low]', 'Narrow Treatment')
        elif 'tmp_treatment[T.high]' in cov_name:
            clean_cov_name = cov_name.replace('tmp_treatment[T.high]', 'Broad Treatment')
        else:
            clean_cov_name = cov_name
        cov_map[cov_name] = clean_cov_name.replace('_', ' ')
    sg.rename_covariates(cov_map=cov_map)
    sg.custom_note_label("")

    tex = sg.render_latex(only_tabular=True)
    with open(fp, 'w') as f:
        f.writelines(tex)


def reduced_form_analysis(tv, ov, d, covariates, fn, behaviors, other_controls, xlabel,
                          redefine_treatment=False, legend_loc=None, weight_col='weight_full'):
    sg_models = []
    sg_models_details = {}
    lines = {
        'Control: Other Treatment': [],
        'Behavior': [],
        'Number Controls': [],
        'Number Treated': []
    }
    results = {}
    coefs = None
    for other_control in other_controls:
        sg_models_details[other_control] = {}
        rf_coefs = []
        # fts = {}
        for el in behaviors:
            log.info('OV: {0}, TV: {1}, Other Control: {2}, Behavior: {3}'.format(ov.format(el), tv, other_control, el))
            tmp = d.copy()

            if redefine_treatment:
                # in this case we need to redefine treatment to be 'tv' if info_pos < ov_pos
                treatment_col = 'surveyresponseinformation_pos'
                outcome_block = cleancol2block[ov.format(el)]
                outcome_col = outcome_block + '_pos'
                tmp['tmp_treatment'] = np.nan
                tmp.loc[tmp[treatment_col] < tmp[outcome_col], 'tmp_treatment'] = tmp['level']
                tmp.loc[tmp[treatment_col] >= tmp[outcome_col], 'tmp_treatment'] = 'control'
                tmp = tmp.loc[~pd.isna(tmp[treatment_col]) & ~pd.isna(tmp[outcome_col]) & ~pd.isna(tmp.level)]

                # also need to redefine treatment
                for cov in covariates:
                    clean_cov = clean_covariate(cov).replace('_pre_both', '').replace('_pre', '')
                    if clean_cov in ['period']:
                        continue
                    cov_block = cleancol2block[clean_cov.replace('_imputed', '')]
                    cov_col = cov_block + '_pos'
                    if '_pre_both' in cov:
                        ixs = (tmp[cov_col] < tmp[treatment_col]) & (tmp[cov_col] < tmp[outcome_col])
                        tmp.loc[ixs, cov] = tmp.loc[ixs, clean_cov]
                    else:
                        ixs = tmp[cov_col] < tmp[treatment_col]
                        tmp.loc[ixs, cov] = tmp.loc[ixs, clean_cov]
            else:
                tmp['tmp_treatment'] = tmp[tv].copy()
            if other_control:
                tmp = tmp.loc[(tmp.tmp_treatment != 'control')].copy()
                tmp.loc[tmp.raw_behavior != el, 'tmp_treatment'] = 'control'
            else:
                # this filters on raw behavior b/c when treatment is re-defined, behavior==control is meaningless.
                tmp = tmp.loc[(tmp.raw_behavior == el) | (tmp.tmp_treatment == 'control')].copy()

            tmp = tmp.loc[~pd.isna(tmp[ov.format(el)])]
            tmp = tmp.loc[~pd.isna(tmp['tmp_treatment'])].copy()
            log.info(tmp.groupby(['behavior', 'tmp_treatment']).id.count())

            covariates = [c for c in covariates if
                          len(tmp.loc[(tmp.tmp_treatment != 'control') & (tmp.tmp_treatment != 0), clean_covariate(c)].unique()) > 1
                          and len(tmp.loc[(tmp.tmp_treatment == 'control') | (tmp.tmp_treatment == 0), clean_covariate(c)].unique()) > 1]

            tmp_coefs, tmp_mod, reg_df = reduced_form_analysis_single(d=tmp, name=el, tv='tmp_treatment', ov=ov.format(el),
                                                                      covariates=covariates, weight_col=weight_col)
            rf_coefs.append(tmp_coefs)

            # now add for stargazer
            sg_models.append(tmp_mod)
            sg_models_details[other_control][el] = tmp_mod
            if other_control:
                lines['Control: Other Treatment'].append('X')
            else:
                lines['Control: Other Treatment'].append('')
            lines['Behavior'].append(el)
            lines['Number Controls'].append(np.sum((reg_df.tmp_treatment == 'control') | (reg_df.tmp_treatment == 0)))
            lines['Number Treated'].append(np.sum((reg_df.tmp_treatment != 'control') & (reg_df.tmp_treatment != 0)))

        coefs = pd.concat(rf_coefs, axis=0)
        if other_control:
            fp = os.path.join(fig_paths[weight_col], fn.replace('.', '_othercontrol.'))
        else:
            fp = os.path.join(fig_paths[weight_col], fn)
        log.info(coefs)
        coefs.group = [group_map[el] if el in group_map else el for el in coefs.group]
        ChartTools.plot_coefs(coefs=coefs, fn=fp, title='', pickle_fig=False, offset_amt=0.1, color_map=color_map, legend_loc=legend_loc, xlabel=xlabel)
        results[other_control] = coefs

    cov_order = ['Intercept'] if 'Intercept' in list(coefs.term) else [] + [el for el in np.unique(coefs.term)]
    save_tables(fp=os.path.join(fig_paths[weight_col], fn.replace('.pdf', '.tex')), sg_models=sg_models, lines=lines, covariate_order=cov_order)
    return sg_models_details, results


def reduced_form_analysis_hte(tv, ov, d, covariates, fn, hte_var, xlabel, weight_col, treatment_filter=None, other_controls=(True, False), xlim=None):
    raw_hte_var = hte_var
    log.info('HTE: {0}'.format(hte_var))
    assert ov == 'future_{0}', 'Not implemented yet'
    results = {}
    for other_control in other_controls:
        results[other_control] = {}
        for behavior in ['vaccine']:
            hte_var = raw_hte_var.format(behavior)
            rf_coefs = []
            tmp = d.copy()
            if other_control:
                tmp = tmp.loc[tmp[tv] != 'control'].copy()
                tmp['tmp_treatment'] = tmp[tv]
                tmp.loc[tmp.behavior != behavior, 'tmp_treatment'] = 'control'
            else:
                tmp = tmp.loc[tmp.behavior.isin([behavior, 'control'])].copy()
                tmp['tmp_treatment'] = tmp[tv]
            if not isinstance(treatment_filter, type(None)):
                tmp = tmp.loc[tmp.treatment.isin(treatment_filter + ['control'])]

            tmp = tmp.loc[~pd.isna(tmp[ov.format(behavior)])].copy()
            tmp = tmp.loc[~pd.isna(tmp[tv])].copy()

            if behavior == 'dist':
                # filter out data after we sunsetted dist, this avoids singular matrix problems
                tmp = tmp.loc[tmp.survey_response_information_version <= 3].copy()

            average_coefs, mod, _ = reduced_form_analysis_single(d=tmp.copy(), name=behavior, tv='tmp_treatment', ov=ov.format(behavior), covariates=covariates, weight_col=weight_col)
            average_coefs.index = ['Average'] * len(average_coefs)

            vals = [el for el in tmp[hte_var].unique() if not pd.isna(el)]
            vals = np.sort(vals)

            sg_models = [mod]
            sg_labels = ['Average']
            num_treatments = len(tmp.tmp_treatment.unique()) - 1
            num_params = num_treatments * len(vals)
            vcov = np.zeros((num_params, num_params))
            params = np.zeros(num_params)
            param_names = []
            last_ix = 0
            log.info('OV: {0}, TV: {1}, Other Control: {2}, Behavior: {3}, HTE: {4}'.format(ov.format(behavior), tv, other_control, behavior, hte_var))
            for val in vals:
                log.info(val)
                assert not pd.isna(val)
                tmp_val = tmp.loc[tmp[hte_var] == val].copy()
                index_name = val
                tmp_coefs, mod, _ = reduced_form_analysis_single(d=tmp_val, name=behavior, tv='tmp_treatment', ov=ov.format(behavior), covariates=covariates, weight_col=weight_col)
                tmp_coefs.index = [index_name] * len(tmp_coefs)
                rf_coefs.append(tmp_coefs)
                sg_models.append(mod)
                sg_labels.append(val)

                # add to params and vcov
                ixs = [el in tmp_coefs.term.to_list() for el in mod.params.index]
                param_names += tmp_coefs.term.to_list()
                assert np.sum(ixs) == num_treatments
                params[last_ix:(last_ix + num_treatments)] = mod.params[ixs]
                vcov[last_ix:(last_ix + num_treatments), last_ix:(last_ix + num_treatments)] = mod.cov_HC2[np.ix_(ixs, ixs)]
                last_ix += np.sum(ixs)

            # now run the joint test
            unique_params = np.unique(param_names)
            num_restrictions = num_params - num_treatments
            r = np.zeros((num_restrictions, len(params)))
            r_row = 0
            for p_name in unique_params:
                first_param_name_index = None
                for pix, p in enumerate(param_names):
                    if p != p_name:
                        continue
                    if isinstance(first_param_name_index, type(None)):
                        first_param_name_index = pix
                    else:
                        r[r_row, first_param_name_index] = 1
                        r[r_row, pix] = -1
                        r_row += 1
            rp = r @ params
            statistic = rp.T @ np.linalg.inv(r @ vcov @ r.T) @ rp
            p_val = 1 - scipy.stats.chi2.cdf(statistic, num_restrictions)
            log.info('Joint test that treatments are equal (highs==highs & lows==lows): Statistic {0}, p-val: {1}'.format(statistic, p_val))

            coefs = pd.concat(rf_coefs, axis=0)
            coefs['sort_order'] = [el for el in coefs.index]
            if hte_var == 'country_pre':
                coefs['sort_order'] = ['{0}_{1}'.format(coefs.loc[el, 'Estimate'].mean(), coefs.iloc[ix]['group']) for ix, el in enumerate(coefs.index)]
            coefs = coefs.sort_values('sort_order')
            coefs = pd.concat([coefs] + [average_coefs], axis=0)
            coefs.group = [group_map[el] if el in group_map else el for el in coefs.group]

            # log.info(coefs)
            fp = os.path.join(fig_paths[weight_col], fn.format(hte_var, behavior))
            if other_control:
                fp = fp.replace('.', '_othercontrol.')

            clean_hte_var = hte_var.replace('_pre', '').replace('_both', '')
            if clean_hte_var in label_map:
                coefs.index = [label_map[clean_hte_var][el] if el in label_map[clean_hte_var] else el
                               for el in coefs.index]
            ChartTools.plot_coefs(coefs=coefs, fn=fp, title='', pickle_fig=False, offset_amt=0.1, color_map=color_map, xlabel=xlabel, xlim=xlim)

            # save table
            cov_order = ['Intercept'] if 'Intercept' in list(coefs.term) else [] + [el for el in np.unique(coefs.term)]
            save_tables(fp=os.path.join(fig_paths[weight_col], fn.format(hte_var, behavior).replace('.pdf', '.tex')), sg_models=sg_models, lines={}, dependent_variable_name=sg_labels, covariate_order=cov_order)
            results[other_control][behavior] = coefs
    return results


def plot_outcome_distribution(d, ov, tv, focal_behavior, covariates, weight_col='weight_full'):
    covariates = [el for el in covariates if 'norms_dist' not in el]
    ov = ov.format(focal_behavior)
    # filter people missing treatment or outcome
    d = d.loc[~pd.isna(d[ov])].copy()
    d = d.loc[~pd.isna(d[tv])].copy()
    d = d.loc[~pd.isna(d['behavior'])].copy()

    d['tmp_treatment'] = d.treatment
    d.loc[~d.behavior.isin([focal_behavior, 'control']), 'tmp_treatment'] = 'other_behavior'
    log.info(d.tmp_treatment.value_counts())

    tmp_treatment2col = {
        'control': 'Control',
        'other_behavior': 'Other Behavior',
        'high': 'Broad',
        'low': 'Narrow'
    }

    # now group by tmp_treatment and outcome
    agg = pd.DataFrame(index=np.sort(d[ov].unique()), columns=['Control', 'Other Behavior', 'Broad', 'Narrow'])
    for t in tmp_treatment2col:
        tmp = d.loc[d.tmp_treatment == t].copy()
        for v in agg.index:
            agg.loc[v, tmp_treatment2col[t]] = np.average(tmp[ov] == v, weights=tmp[weight_col])

    # replace index w/ text
    if ov in label_map:
        agg.index = [label_map[ov][el] for el in agg.index]

    if not os.path.exists(os.path.join(fig_paths[weight_col], 'distribution_plots')):
        os.makedirs(os.path.join(fig_paths[weight_col], 'distribution_plots'))
    fn = os.path.join(fig_paths[weight_col], 'distribution_plots', 'ov{0}_tv{1}_behavior{2}.pdf'.format(ov, tv, focal_behavior))
    fig, ax = plt.subplots()
    cs = []
    other_c_ix = 0
    for el in agg.columns:
        if el in color_map:
            cs.append(color_map[el])
        else:
            cs.append(other_colors[other_c_ix])
            other_c_ix += 1
    agg.plot(ax=ax, kind='bar', color=cs)
    plt.xticks(rotation=0)
    ChartTools.save_show_plot(fig=fig, fn=fn, show_graph=False, pickle_fig=True)
    log.info(agg)

    # also do this with a model
    agg2 = pd.DataFrame(index=np.sort(d[ov].unique()), columns=[])
    agg2_errors = pd.DataFrame(index=np.sort(d[ov].unique()), columns=[])
    for v in agg2.index:
        d['tmp_outcome'] = pd.Series(d[ov] == v).astype(int)

        mod = RegressionTools.lm_lin_manual(df=d.copy(), covariates=covariates, ov='tmp_outcome', tv='tmp_treatment',
                                            se_type='HC2', sm_fit_collinear=True, no_intercept=True, weights=d[weight_col])
        agg2.loc[v, 'Control'] = mod.params['tmp_treatment[control]']
        agg2_errors.loc[v, 'Control'] = mod.bse['tmp_treatment[control]'] * 1.96
        agg2.loc[v, 'Other Behavior'] = mod.params['tmp_treatment[other_behavior]']
        agg2_errors.loc[v, 'Other Behavior'] = mod.bse['tmp_treatment[other_behavior]'] * 1.96
        agg2.loc[v, 'Broad'] = mod.params['tmp_treatment[high]']
        agg2_errors.loc[v, 'Broad'] = mod.bse['tmp_treatment[high]'] * 1.96
        agg2.loc[v, 'Narrow'] = mod.params['tmp_treatment[low]']
        agg2_errors.loc[v, 'Narrow'] = mod.bse['tmp_treatment[low]'] * 1.96
    # replace index w/ text
    if ov in label_map:
        lm = label_map[ov].copy()
        for key in lm:
            if lm[key] == 'No, definitely not':
                lm[key] = 'No,\ndefinitely not'
            elif lm[key] == 'Yes, definitely':
                lm[key] = 'Yes,\ndefinitely'
        agg2.index = [lm[el] for el in agg2.index]
        agg2_errors.index = [lm[el] for el in agg2_errors.index]

    fig, ax = plt.subplots()
    agg2.plot(ax=ax, kind='bar', color=cs, yerr=agg2_errors)
    plt.xticks(rotation=0)
    fn = os.path.join(fig_paths[weight_col], 'distribution_plots', 'withcontrols_ov{0}_tv{1}_behavior{2}.pdf'.format(ov, tv, focal_behavior))
    agg2.to_csv(fn.replace('.pdf', '.csv'), sep=',')
    ChartTools.save_show_plot(fig=fig, fn=fn, show_graph=False, pickle_fig=True)
    log.info(agg2)


def plot_distribution_of_screens(df):
    df = df.copy()

    def screens_between_treatment_control(do):
        treatment_bo = do['block_order']['surveyresponseinformation']
        outcome_bo = do['block_order']['futureactions']
        num_pages = 0
        if treatment_bo < outcome_bo:
            num_pages = -1  # the masks/dist outcome is one page before vaccine outcome
        start_block = min(treatment_bo, outcome_bo)
        end_block = max(treatment_bo, outcome_bo)
        for b in do['block_order']:
            if do['block_order'][b] <= start_block or do['block_order'][b] >= end_block:
                continue
            if treatment_bo < outcome_bo:
                num_pages -= block2screens[b]
            else:
                num_pages += block2screens[b]
        return num_pages

    df['num_screens'] = df.display_order.apply(screens_between_treatment_control)

    # now plot
    fig, ax = plt.subplots()
    nbins = df.num_screens.max() - df.num_screens.min() + 2
    df.num_screens.hist(ax=ax, grid=False, bins=nbins, density=True, color=other_colors[0])
    plt.xlabel('Treatment Screen Number - Outcome Screen Number')
    fn = os.path.join(fig_paths['weight_full'], 'distribution_plots', 'screen_gap_distribution.pdf')
    ChartTools.save_show_plot(fig=fig, fn=fn, show_graph=False, pickle_fig=True)

    # plot for those separated by at least one block
    fig, ax = plt.subplots()
    mv = df.num_screens.abs().max()
    bins = np.linspace(-mv, mv, 2*mv + 1)
    df.loc[df.treatment_outcome_gap > 1].num_screens.hist(ax=ax, grid=False, bins=bins, density=True, color=other_colors[0])
    plt.xlabel('Treatment Screen Number - Outcome Screen Number')
    fn = os.path.join(fig_paths['weight_full'], 'distribution_plots', 'screen_gap_distribution_robustness.pdf')
    ChartTools.save_show_plot(fig=fig, fn=fn, show_graph=False, pickle_fig=True)


def finalize_figs():
    # merge baseline belief partition & vaccine accept HTEs for the figure in the paper
    hte_vars = ['baseline_belief_partition_vaccine_pre_both', 'vaccine_accept_pre']
    coefs = pd.DataFrame()
    hte_prefix = {'baseline_belief_partition_vaccine_pre_both': 'Norms: ', 'vaccine_accept_pre': 'Vaccine: '}
    for v in hte_vars:
        tmp = pd.read_table(os.path.join(fig_paths['weight_full'], 'hte/treatment/hte{0}_behavior{1}.csv'.format(v, 'vaccine')), sep=',', index_col=0)
        tmp['hte_var'] = v
        tmp.index = [hte_prefix[v] + el if el.lower() != 'average' else el for el in tmp.index]
        if v != hte_vars[-1]:
            tmp = tmp.loc[[True if el.lower() != 'average' else False for el in tmp.index]]
        coefs = pd.concat([coefs, tmp])
    ChartTools.plot_coefs(coefs=coefs, fn=os.path.join(fig_paths['weight_full'], 'hte/treatment/hte_merged_for_paper.pdf'),
                          title='', xlabel='Effect on vaccine acceptance scale', color_map=color_map, legend_loc='upper left',
                          triangle_points=['Average'], offset_amt=0.2)


def weight_plots(df):
    countries = [el for el in df.country.unique() if not pd.isna(el)]
    aggs = {'Weighted': pd.DataFrame(index=countries), 'Unweighted': pd.DataFrame(index=countries)}

    # filter to only include males or females
    df = df.loc[df.weight_full > 0].copy()
    df['female'] = (df.gender == 2).apply(int)

    for c in countries:
        tmp = df.loc[df.country == c]

        # weighted estimates
        mod = smf.wls('female ~ 1', data=tmp, weights=tmp.weight_full).fit()
        aggs['Weighted'].loc[c, 'Estimate'] = mod.params.iloc[0]
        aggs['Weighted'].loc[c, 'SE'] = mod.bse.iloc[0]

        # unweighted estimates
        mod = smf.wls('female ~ 1', data=tmp, weights=tmp.unweighted).fit()
        aggs['Unweighted'].loc[c, 'Estimate'] = mod.params.iloc[0]
        aggs['Unweighted'].loc[c, 'SE'] = mod.bse.iloc[0]
    for k in aggs:
        aggs[k]['group'] = k
    country_order = list(aggs['Unweighted'].sort_values('Estimate').index)
    aggs = pd.concat([aggs['Weighted'], aggs['Unweighted']], axis=0)
    aggs = aggs.loc[country_order]
    ChartTools.plot_coefs(coefs=aggs, fn=os.path.join(raw_fig_path, 'weight_fig_gender.pdf'), pickle_fig=False, line_loc=0.5, xlabel='Share female', markersize=3)


def treatment_effects_by_number(df, covariates, weight_col='weight_full'):
    fp = os.path.join(fig_paths[weight_col], 'treatment_number_regs')
    if not os.path.exists(fp):
        os.makedirs(fp)

    for level in ['high', 'low']:
        tmp = df.loc[df.behavior.isin(['control', 'vaccine']) & df.treatment.isin(['control', level])].copy()
        tmp['treatment_number'] = tmp.apply(axis=1, func=lambda row: row['{0}_value_vaccine'.format(level)])
        tmp['treatment_number_bin'] = tmp.treatment_number.apply(lambda v: ' ' + str(v // 20 * 20))
        # tmp.loc[tmp.treatment_number.apply(lambda n: False if n == 'control' else int(n) <= 40), 'treatment_number_bin'] = 'value_40'

        tmp['treatment_binary'] = tmp.treatment.apply(lambda t: 'treated' if t != 'control' else 'control')
        cols2keep = ['behavior', 'treatment_binary', 'treatment_number_bin', 'future_vaccine', weight_col, 'vaccine_accept'] + [clean_covariate(el) for el in covariates]
        tmp = tmp.drop(columns=[el for el in tmp.columns if el not in cols2keep])
        reduced_form_analysis_hte(
            tv='treatment_binary', ov='future_{0}', d=tmp, covariates=[el for el in covariates if 'country' not in el], fn='treatment_number_regs/main_effects_by_treatment_number_{0}.pdf'.format(level),
            hte_var='treatment_number_bin', treatment_filter=None, xlabel='Treatment effect on five point scale', other_controls=[False],
            weight_col=weight_col, xlim=(-0.1, 0.15)
        )
        reduced_form_analysis_hte(
            tv='treatment_binary', ov='future_{0}', d=tmp.loc[tmp.vaccine_accept == 1], covariates=[el for el in covariates if 'country' not in el], fn='treatment_number_regs/unsure_main_effects_by_treatment_number_{0}.pdf'.format(level),
            hte_var='treatment_number_bin', treatment_filter=None, xlabel='Treatment effect on five point scale', other_controls=[False],
            weight_col=weight_col, xlim=(-0.1, 0.25)
        )


class RegressionTools:
    @staticmethod
    def lm_robust(df, formula, clusters=None, weights=None, missing='raise', se_type='HC2', sm_fit_collinear=False):
        if isinstance(weights, type(None)):
            weights = 1
        else:
            assert len(df) == len(weights)
        mod = smf.wls(
            formula=formula,
            data=df,
            weights=weights,
            missing=missing
        )
        if isinstance(clusters, type(None)) and not sm_fit_collinear:
            fit = mod.fit(cov_type=se_type)
        elif not sm_fit_collinear:
            fit = mod.fit(cov_type='cluster', cov_kwds={'groups': df[clusters]})
        elif isinstance(clusters, type(None)):
            fit = mod._fit_collinear(cov_type=se_type)
        else:
            fit = mod._fit_collinear(cov_type='cluster', cov_kwds={'groups': df[clusters]})

        # print wald test
        if 'tmp_treatment[T.high]' in fit.params and 'tmp_treatment[T.low]' in fit.params:
            log.info(fit.wald_test('tmp_treatment[T.high]=tmp_treatment[T.low]'))
        return fit

    @staticmethod
    def clean_covariate(s):
        return s.replace('c(', '').replace(')', '').replace(' ', '').replace('factor(', '').replace('C(', '')

    @staticmethod
    def lin_preprocessor(treatment, outcome, covariates, d, extra_cols, weights, dont_center=False):
        d = d.copy()

        # get covariate design matrix
        #     d_mat =
        if len(covariates) > 0:
            xs = patsy.dmatrix('+'.join(covariates), data=d, return_type='dataframe')
            xs = xs[[el for el in xs.columns if el != 'Intercept']]
            if not dont_center:
                xs = xs - np.average(xs, axis=0, weights=weights)
        else:
            xs = pd.DataFrame(index=d.index)
        formula = '{0} ~ 1 + {1}'.format(outcome, treatment)
        for c in list(xs.columns):
            # all covariates should be interacted w/ treatment
            clean_c = c.split('[')[0]
            interact = True
            assert clean_c in covariates

            # now generate new c
            new_col = c.replace('C(', '').replace(')', '__').replace('[T.', 'cat_').replace('[', 'cat_').replace(']', '').replace('.', '_')
            xs = xs.rename(columns={c: new_col})
            if interact:
                formula += '+ {1}:{0} + {0}'.format(new_col, treatment)
            else:
                formula += '+ {0}'.format(new_col)
        old_len = len(d)
        for el in [treatment, outcome] + [el for el in extra_cols if not isinstance(el, type(None))]:
            clean_c = clean_covariate(el)
            xs[clean_c] = d[clean_c]
        d = xs.copy()
        assert len(d) == old_len, 'Old len: {0}, new len: {1}'.format(old_len, len(d))
        return d, formula

    @staticmethod
    def lm_lin_manual(df, covariates, ov, tv, weights=None, clusters=None, se_type=None,
                      dont_center=False, sm_fit_collinear=False, no_intercept=False):
        # pre-process data
        # do this before conditioning on country so we are post-stratifying using same distribution of x's w/n each country
        reg_df, formula = RegressionTools.lin_preprocessor(
            treatment=tv,
            outcome=ov,
            covariates=covariates,
            d=df,
            extra_cols=[clusters],
            dont_center=dont_center,
            weights=weights
        )
        assert len(reg_df.columns) == len(np.unique(reg_df.columns))

        if not dont_center:
            covariates = [el for el in formula.split('~')[1].split('+') if ':' not in el and tv not in el]
            covariates = [el.replace(' ', '') for el in covariates]
            covariates = [el for el in covariates if el in reg_df.columns]
            if len(covariates) > 0:
                means = np.average(reg_df[covariates], axis=0, weights=weights)
                abs_means = np.abs(means)
                assert np.max(abs_means) < 0.001, abs_means

        if no_intercept:
            formula = formula.replace('~', '~ 0 + ')
            formula = formula.replace(' 1 +', '')

        # now estimate model
        mod = RegressionTools.lm_robust(
            df=reg_df.copy(),
            formula=formula,
            clusters=clusters,
            se_type=se_type,
            weights=weights,
            sm_fit_collinear=sm_fit_collinear
        )
        return mod


class ChartTools:
    @staticmethod
    def save_show_plot(fig: plt.figure, fn: str, show_graph: bool, pickle_fig: bool = True, tight_layout: bool = True):
        dir_name = os.path.dirname(fn)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if pickle_fig:
            pickle_name = os.path.splitext(fn)[0] + '.p'
            with open(pickle_name, 'wb') as f:
                pickle.dump(fig, f)

        if tight_layout:
            try:
                fig.tight_layout()
            except ValueError:
                pass
        fig.savefig(fn, transparent=True)
        if show_graph:
            plt.show()
        plt.close()

    @staticmethod
    def plot_coefs(coefs: pd.DataFrame, fn: str, title: str = '', ylabel: str = '', xlabel: str = '', pickle_fig: bool = True, offset_amt=0.1,
                   label_map=None, color_map=None, line_loc=0, xlim=None, legend_loc=None, triangle_points=(), markersize=6):
        cols = list(coefs.columns)

        coefs.to_csv(fn.split('.')[0] + '.csv')

        if isinstance(label_map, type(None)):
            label_map = {}

        if 'group' not in cols:
            coefs['group'] = 1
        has_groups = len(coefs['group'].value_counts()) > 1

        cols = list(coefs.columns)
        assert 'Estimate' in cols
        assert 'SE' in cols
        assert 'group' in cols

        coefs['ci'] = 1.96 * coefs.SE

        values = []
        for el in coefs.index:
            if el not in values:
                values.append(el)
        num_values = len(values)

        # calculate axis offsets
        groups = []
        for el in coefs['group']:
            if el not in groups and not pd.isna(el):
                groups.append(el)
        num_groups = len(groups)
        axis_pos = np.arange(num_values)
        if num_groups % 2 == 0:
            left_most_point = -num_groups / 2 * offset_amt + offset_amt / 2
        else:
            left_most_point = -(num_groups - 1) / 2 * offset_amt
        offsets = {}
        for ix, group in enumerate(groups):
            offsets[group] = left_most_point + ix * offset_amt

        middle_points = {}
        for ix, v in enumerate(values):
            middle_points[v] = axis_pos[ix]

        fig, ax = plt.subplots()
        for ix, group in enumerate(groups):
            if pd.isna(group):
                continue

            tmp = coefs.loc[coefs.group == group].copy()
            axis_locs = [middle_points[el] + offsets[group] for el in tmp.index]
            tmp.index = [str(el) for el in tmp.index]
            c = None
            if not isinstance(color_map, type(None)) and group in color_map:
                c = color_map[group]

            triangle_ixs = [True if ii in triangle_points else False for ii in tmp.index]
            circle_ixs = [False if ii in triangle_points else True for ii in tmp.index]
            ax.errorbar(x=tmp.loc[circle_ixs].Estimate, y=[axis_locs[ii] for ii in range(len(axis_locs)) if circle_ixs[ii]],
                        xerr=tmp.loc[circle_ixs].ci, fmt='o', label=group, color=c, markersize=markersize)
            ax.errorbar(x=tmp.loc[triangle_ixs].Estimate, y=[axis_locs[ii] for ii in range(len(axis_locs)) if triangle_ixs[ii]],
                        xerr=tmp.loc[triangle_ixs].ci, fmt='^', color=c, markersize=markersize)

        if not isinstance(line_loc, type(None)):
            ax.axvline(x=line_loc, ymin=0, ymax=1, color='k')

        if not isinstance(xlim, type(None)):
            plt.xlim(xlim[0], xlim[1])

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)

        # add label text
        ax.set_yticks(axis_pos)
        ax.set_yticklabels([label_map[el] if el in label_map else el for el in values])

        if has_groups:
            ax.legend(loc=legend_loc)
        ChartTools.save_show_plot(fig=fig, fn=fn, show_graph=False, pickle_fig=pickle_fig, tight_layout=True)


def main():
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/covid_survey_responses_numeric.txt.gz')
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/random_demo_data.txt.gz')
    full_df = pd.read_table(fn, sep='\t', low_memory=False)

    # filter to eligible for treatment
    full_df = full_df.loc[(full_df.eligible_for_information == 1) & (full_df.survey_type == 'waves')]
    full_df['behavior'] = full_df.survey_information_behavior
    full_df['raw_behavior'] = full_df.behavior.copy()
    full_df['level'] = full_df.survey_information_level
    full_df['value'] = full_df.survey_information_value

    run_all = False
    other_control_preferred_model = False
    df = full_df.copy()
    full_df = None
    df.display_order = df.display_order.apply(json.loads)
    df = normalize_weights(df)
    df = process_data(df)
    df['unweighted'] = 1

    #################
    # baseline figs #
    #################
    if run_all:
        plot_treatments(df)

    #############################
    # Data Cleaning & Filtering #
    #############################

    # Note build_long_data conditions on being eligible for treatment and
    # conditions on reaching both treatment and control
    df['intent_to_treat'] = df.apply(func=lambda row: 'control' if row['futureactions_pos'] < row['surveyresponseinformation_pos'] else 'treated', axis=1)
    df['treatment'] = df.apply(func=lambda row: 'control' if row['futureactions_pos'] < row['surveyresponseinformation_pos'] else row['level'], axis=1)
    df['any_treatment'] = df.treatment.apply(lambda vv: vv if pd.isna(vv) else ('treated' if vv != 'control' else 'control'))
    df['weight_full'] = df.weight_full_survey
    df['period'] = df.wave.apply(lambda vv: 'wave' + str(int(vv))) + '_' + df.survey_response_information_version.apply(lambda vv: 'infoV' + str(int(vv)))
    df.loc[df.intent_to_treat == 'control', 'behavior'] = 'control'
    # df['baseline_belief_partition_imputed'] = df.baseline_belief_partition.apply(str)

    df['treatment_outcome_gap'] = np.abs(df.futureactions_pos - df.surveyresponseinformation_pos)
    df['exists_treatment_outcome_gap'] = df.treatment_outcome_gap > 1
    weight_plots(df)

    df = add_baseline_belief_partition(df)

    #######################
    # run balance checks  #
    #######################
    if run_all:
        if not os.path.exists(os.path.join(raw_fig_path, 'randomization_checks')):
            os.makedirs(os.path.join(raw_fig_path, 'randomization_checks'))

        log.info('============= Randomization Checks =====================')
        full_rand_test = test_randomization(d=df, treatment_indicator='intent_to_treat', sample_name='full')
        final_rand_test = test_randomization(d=df.loc[~pd.isna(df.weight_full)], treatment_indicator='treatment', sample_name='analysis')
        robust_rand_test = test_randomization(d=df.loc[~pd.isna(df.weight_full) & df.exists_treatment_outcome_gap], treatment_indicator='treatment', sample_name='robust')
        rand_test = pd.DataFrame(index=['Full', 'Final'], columns=['p-val', 'Treated Share', 'Control Share'])
        rand_test.loc['Full'] = [full_rand_test['p'], full_rand_test['treated'], full_rand_test['control']]
        rand_test.loc['Final'] = [final_rand_test['p'], final_rand_test['treated'], final_rand_test['control']]
        rand_test.loc['Robust'] = [robust_rand_test['p'], robust_rand_test['treated'], robust_rand_test['control']]
        rand_test.to_latex(
            os.path.join(raw_fig_path, 'randomization_checks', 'randomization_tests.tex'),
            float_format='{:0.3f}'.format
        )

        # now test within treatment randomizations
        within_tests = pd.DataFrame(0.0, index=['Final', 'Robust'], columns=['Vaccine', 'Masks', 'Distancing'])
        for b in df.behavior.unique():
            tmp = df.loc[(df.intent_to_treat != 'control') & ~pd.isna(df.weight_full)].copy()
            if pd.isna(b) or b == 'control':
                continue
            tmp['tmp_treatment'] = 'control'
            tmp.loc[tmp.behavior == b, 'tmp_treatment'] = 'treated'
            if b == 'dist':
                tmp = tmp.loc[tmp.wave < 12].copy()
                wave2share = dict((el, 1/3) for el in tmp.wave.unique())
                col = 'Distancing'
            elif b == 'masks':
                wave2share = dict((el, 1/3) for el in tmp.wave.unique())
                col = 'Masks'
            else:
                assert b == 'vaccine'
                wave2share = dict((el, 1/3) if el < 12 else (el, 2/3) for el in tmp.wave.unique())
                col = 'Vaccine'
            rand_test = test_randomization_within(d=tmp, treatment_indicator='tmp_treatment', sample_name=b, wave2proportion=wave2share)
            rand_test_robust = test_randomization_within(d=tmp.loc[tmp.exists_treatment_outcome_gap], treatment_indicator='tmp_treatment', sample_name=b, wave2proportion=wave2share)
            within_tests.loc['Final', col] = rand_test['p']
            within_tests.loc['Robust', col] = rand_test_robust['p']
        within_tests.to_latex(
            os.path.join(raw_fig_path, 'randomization_checks', 'randomization_tests_within.tex'),
            float_format='{:0.3f}'.format
        )

        log.info('\n\n============= Balance Checks ===========================')
        df['equal_weight'] = 1
        if pd.isna(df.weight_full).sum() > 0:
            test_balance(d=df, sample_name='full', weights='equal_weight')
        test_balance(d=df.loc[~pd.isna(df.weight_full)], sample_name='final', weights='weight_full')
        test_balance_within_treatments(d=df.loc[~pd.isna(df.weight_full) & (df.weight_full > 0)], sample_name='final', weights='weight_full')

        log.info('=============== Attrition checks ==========================')
        control_completes = df.loc[df.intent_to_treat == 'control'].finished
        treated_completes = df.loc[df.intent_to_treat == 'treated'].finished
        log.info(df.groupby('id').first().groupby('intent_to_treat').finished.mean())
        stat, p_val = scipy.stats.ttest_ind(control_completes, treated_completes, equal_var=False)
        log.info('Test of equal completion ITT: {0}'.format(p_val))

        mask_completes = df.loc[(df.intent_to_treat == 'treated') & (df.behavior == 'masks')].finished
        vacc_completes = df.loc[(df.intent_to_treat == 'treated') & (df.behavior == 'vaccine')].finished
        dist_completes = df.loc[(df.intent_to_treat == 'treated') & (df.behavior == 'dist')].finished
        log.info(df.loc[df.intent_to_treat == 'treated'].groupby('behavior').finished.mean())
        vm_stat, vm_p_val = scipy.stats.ttest_ind(vacc_completes, mask_completes, equal_var=False)
        vd_stat, vd_p_val = scipy.stats.ttest_ind(vacc_completes, dist_completes, equal_var=False)
        log.info('Test of equal completion w/n treatment (V vs. M): {0}'.format(vm_p_val))
        log.info('Test of equal completion w/n treatment (V vs. D): {0}'.format(vd_p_val))

    #################################
    # covariates to use in analysis #
    #################################
    # df['country_cleaned_pre_both'] = df.country_pre.apply(lambda c: c.replace(' ', ''))
    # df['country_cleaned_pre'] = df.country_pre.apply(lambda c: c.replace(' ', ''))
    df['future_mediator_position'] = df['futureactions_pos'].apply(str) + '_' + df['distancingfamiliarity,importance&norms_pos'].apply(str)

    dum_covariates = ['vaccine_accept']
    linear_covariates = ['norms_dist', 'norms_vaccine', 'norms_masks',
                         'prevention_distancing', 'prevention_mask', 'effect_mask',
                         'distancing_importance']

    dum_covariates = ['C({0}_pre_both_imputed)'.format(el) for el in dum_covariates]
    linear_covariates = [el + '_pre_both_imputed' for el in linear_covariates]
    covariates = dum_covariates + linear_covariates
    covariates += ['C(period)', 'C(country_cov)']
    ###############
    # filter data #
    ###############
    constants['n_start'] = len(df)
    log.info('Filtering data')
    ixs2drop = pd.isna(df.weight_full) | (df.weight_full == 0)
    log.info('Dropping {0}/{1} ids due to not having full weight.'.format(len(df.loc[ixs2drop].id.unique()), len(df.id.unique())))
    df = df.loc[~ixs2drop].copy()

    ixs2drop = pd.isna(df.behavior)
    log.info('Dropping {0}/{1} ids b/c null behavior.'.format(len(df.loc[ixs2drop].id.unique()), len(df.id.unique())))
    df = df.loc[~ixs2drop].copy()

    ixs2drop = pd.isna(df.country)
    log.info('Dropping {0}/{1} ids b/c null country.'.format(len(df.loc[ixs2drop].id.unique()), len(df.id.unique())))
    df = df.loc[~ixs2drop].copy()

    period_thresh = 10
    periods2drop = df.period.value_counts()
    periods2drop = [el for el in periods2drop[periods2drop < period_thresh].index]
    ixs2drop = df.period.isin(periods2drop)
    log.info('Dropping {0}/{1} ids b/c in a period w/ fewer than {2} respondents.'.format(len(df.loc[ixs2drop].id.unique()), len(df.id.unique()), period_thresh))
    df = df.loc[~ixs2drop].copy()
    constants['n_analysis'] = len(df)

    df['country_cov'] = df.country.apply(lambda cc: cc.replace(' ', ''))
    log.info('\n\n')
    #######################
    # manipulation checks #
    #######################
    log.info('Manipulation checks')
    reduced_form_analysis(tv='treatment', ov='norms_{0}', d=df, covariates=covariates,
                          fn='manipulation_checks.pdf', redefine_treatment=True,
                          behaviors=['masks', 'dist', 'vaccine'], other_controls=[True, False],
                          legend_loc='upper left', xlabel='Treatment effect on beliefs about norms')
    log.info('\n\n\n\n')

    ################
    # main effects #
    ################
    df['treatment_outcome_gap'] = np.abs(df.futureactions_pos - df.surveyresponseinformation_pos)
    df['exists_treatment_outcome_gap'] = df.treatment_outcome_gap > 1
    plot_outcome_distribution(d=df, ov='future_vaccine', tv='treatment', focal_behavior='vaccine', covariates=covariates)

    # only run this if not on the demo data. The block order is complicated and hard to simulate so only run this on the real data
    if len(df) > 1000000:
        plot_distribution_of_screens(df=df)

    log.info('Main effects')
    for weight_col in ['weight_full', 'unweighted']:
        ov = 'future_{0}'
        reduced_form_analysis(tv='treatment', ov=ov, d=df, covariates=covariates,
                              fn='main_effects.pdf', redefine_treatment=False,
                              behaviors=['masks', 'dist', 'vaccine'], other_controls=[True, False],
                              legend_loc='upper left', xlabel='Treatment effect on five point scale',
                              weight_col=weight_col)

    # also run conditioning on not having treatment & outcome w/n two blocks of one another
    reduced_form_analysis(tv='treatment', ov=ov, d=df.loc[df.exists_treatment_outcome_gap], covariates=covariates,
                          fn='robustness/main_effects_with_gap.pdf', redefine_treatment=False,
                          behaviors=['masks', 'dist', 'vaccine'], other_controls=[True, False],
                          legend_loc='upper left', xlabel='Treatment effect on five point scale')
    log.info('\n\n\n\n')

    for b in ['masks', 'vaccine', 'dist']:
        col = 'future_{0}'.format(b)
        unique_vals = [el for el in df[col].unique() if not pd.isna(el)]
        for v in unique_vals:
            if v == np.max(unique_vals):
                continue
            new_col = col + '_{0}'.format(int(v))
            df[new_col] = np.nan
            ixs = ~pd.isna(df[col])
            df.loc[ixs, new_col] = pd.Series(df.loc[ixs, col] > v).astype(int)

    weight_col = 'weight_full'
    vaccine_binary_models = {}
    vaccine_binary_models_unsure = {}
    vaccine_binary_models_robustness = {}
    vaccine_binary_models_under = {}
    unsure_baseline = df.loc[df.vaccine_accept == 1].copy()
    robustness_df = df.loc[df.exists_treatment_outcome_gap].copy()
    under_baseline = df.loc[df.baseline_belief_partition_vaccine_pre_both == 'Under'].copy()
    log.info('Distribution Regressions')
    for v in unique_vals:
        if v == np.max(unique_vals):
            continue
        tmp_models, _ = reduced_form_analysis(tv='treatment', ov=ov + '_{0}'.format(int(v)), d=df, covariates=covariates,
                                              fn='main_effects_binary{0}.pdf'.format(int(v)), redefine_treatment=False,
                                              behaviors=['vaccine'], other_controls=[other_control_preferred_model], xlabel='',
                                              weight_col=weight_col
                                              )
        vaccine_binary_models[v] = tmp_models[other_control_preferred_model]['vaccine']

        tmp_models, _ = reduced_form_analysis(tv='treatment', ov=ov + '_{0}'.format(int(v)), d=unsure_baseline, covariates=covariates,
                                              fn='main_effects_binary_unsure_baseline_vaccine_{0}.pdf'.format(int(v)), redefine_treatment=False,
                                              behaviors=['vaccine'], other_controls=[other_control_preferred_model], xlabel='', weight_col=weight_col)
        vaccine_binary_models_unsure[v] = tmp_models[other_control_preferred_model]['vaccine']

        tmp_models, _ = reduced_form_analysis(tv='treatment', ov=ov + '_{0}'.format(int(v)), d=under_baseline, covariates=covariates,
                                              fn='main_effects_binary_under_baseline_vaccine_{0}.pdf'.format(int(v)), redefine_treatment=False,
                                              behaviors=['vaccine'], other_controls=[other_control_preferred_model], xlabel='', weight_col=weight_col)
        vaccine_binary_models_under[v] = tmp_models[other_control_preferred_model]['vaccine']

        tmp_models, _ = reduced_form_analysis(tv='treatment', ov=ov + '_{0}'.format(int(v)), d=robustness_df, covariates=covariates,
                                              fn='robustness/main_effects_binary_vaccine_{0}.pdf'.format(int(v)), redefine_treatment=False,
                                              behaviors=['vaccine'], other_controls=[other_control_preferred_model], xlabel='', weight_col=weight_col)
        vaccine_binary_models_robustness[v] = tmp_models[other_control_preferred_model]['vaccine']

    save_tables(fp=os.path.join(fig_paths[weight_col], 'main_effects_binary_vaccine.tex'),
                sg_models=[vaccine_binary_models[el] for el in range(len(vaccine_binary_models))],
                covariate_order=['Intercept', 'tmp_treatment[T.low]', 'tmp_treatment[T.high]'], lines={},
                dependent_variable_name=['$>$ {0}'.format(label_map['future_vaccine'][el]) for el in range(len(vaccine_binary_models))]
                )
    save_tables(fp=os.path.join(fig_paths[weight_col], 'main_effects_binary_vaccine_unsure_baseline.tex'),
                sg_models=[vaccine_binary_models_unsure[el] for el in range(len(vaccine_binary_models_unsure))],
                covariate_order=['Intercept', 'tmp_treatment[T.low]', 'tmp_treatment[T.high]'], lines={},
                dependent_variable_name=['$>$ {0}'.format(label_map['future_vaccine'][el]) for el in range(len(vaccine_binary_models))]
                )
    save_tables(fp=os.path.join(fig_paths[weight_col], 'main_effects_binary_vaccine_under_baseline.tex'),
                sg_models=[vaccine_binary_models_under[el] for el in range(len(vaccine_binary_models_under))],
                covariate_order=['Intercept', 'tmp_treatment[T.low]', 'tmp_treatment[T.high]'], lines={},
                dependent_variable_name=['$>$ {0}'.format(label_map['future_vaccine'][el]) for el in range(len(vaccine_binary_models))]
                )
    save_tables(fp=os.path.join(fig_paths[weight_col], 'robustness/main_effects_binary_vaccine.tex'),
                sg_models=[vaccine_binary_models_robustness[el] for el in range(len(vaccine_binary_models_unsure))],
                covariate_order=['Intercept', 'tmp_treatment[T.low]', 'tmp_treatment[T.high]'], lines={},
                dependent_variable_name=['$>$ {0}'.format(label_map['future_vaccine'][el]) for el in range(len(vaccine_binary_models))]
                )
    log.info('\n\n\n\n')

    # treatment effects by number
    log.info('Treatment effect by number')
    treatment_effects_by_number(df=df, covariates=covariates)
    log.info('\n\n\n\n')

    ########
    # HTEs #
    ########
    if not os.path.exists(os.path.join(fig_paths[weight_col], 'hte')):
        os.makedirs(os.path.join(fig_paths[weight_col], 'hte'))
        os.makedirs(os.path.join(fig_paths[weight_col], 'hte', 'treatment'))
        os.makedirs(os.path.join(fig_paths[weight_col], 'hte', 'any_treatment'))
        os.makedirs(os.path.join(fig_paths[weight_col], 'hte', 'treatment_nocontrol'))

    df.loc[df.vaccine_accept_pre == 1, 'wave_vaccine_unsure'] = df.loc[df.vaccine_accept_pre == 1, 'wave']

    treatment_filters = {
        'baseline_diff_granular_high_{0}_pre': ['high'],
        'baseline_diff_granular_high_{0}_pre_both': ['high'],
        'baseline_diff_granular_low_{0}_pre': ['low'],
        'baseline_diff_granular_low_{0}_pre_both': ['low']
    }

    ov = 'future_{0}'
    htes = ['gender', 'exists_treatment_outcome_gap', 'vaccine_accept_pre', 'baseline_belief_partition_{0}_pre_both']
    hte_results = {}
    for hte_v in htes:
        log.info('HTE: {0}'.format(hte_v))
        tf = None
        if hte_v in treatment_filters:
            tf = treatment_filters[hte_v]
        hte_results[hte_v] = reduced_form_analysis_hte(
            tv='treatment', ov=ov, d=df, covariates=covariates, fn='hte/treatment/hte{0}_behavior{1}.pdf',
            hte_var=hte_v, treatment_filter=tf, xlabel='Treatment effect on five point scale', weight_col=weight_col
        )

        if hte_v == 'baseline_belief_partition_{0}_pre_both':
            # also run for those w/o whole number baseline responses
            tmp = df.loc[~pd.isna(df.norms_vaccine_pre_both) & (~df.norms_vaccine_pre_both.isin([0, 50, 100])) != 0]
            hte_results[hte_v + '_robust'] = reduced_form_analysis_hte(
                tv='treatment', ov=ov, d=tmp, covariates=covariates, fn='hte/treatment/robust_hte{0}_behavior{1}.pdf',
                hte_var=hte_v, treatment_filter=tf, xlabel='Treatment effect on five point scale',
                weight_col=weight_col
            )
        log.info('\n\n\n\n')
    finalize_figs()

    # write analysis dataset
    df.to_csv(os.path.join(os.path.dirname(fn), 'analysis_data.csv'), index=None)


if __name__ == '__main__':
    main()
