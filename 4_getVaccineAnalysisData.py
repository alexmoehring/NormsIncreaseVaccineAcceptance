import pandas as pd
import os


f = open("waves_snapshot_countries.txt")
lines = f.readlines()
dict_countries = {}
dict_country_params = {}
dict_waves_snapshot = {}

for line in lines:
    line = line.strip()
    line_split = line.split(",")
    dict_countries[line_split[1]] = line_split[0]
    dict_country_params[line_split[0]] = 1
    wave_or_snapshot = line_split[2]
    if wave_or_snapshot in dict_waves_snapshot:
        tmp = dict_waves_snapshot[wave_or_snapshot]
        tmp[line_split[0]] = 1
        dict_waves_snapshot[wave_or_snapshot] = tmp
    else:
        tmp = {}
        tmp[line_split[0]] = 1
        dict_waves_snapshot[wave_or_snapshot] = tmp


def mergeAlreadyVaccinated(vaccine_accept):
    if(vaccine_accept=="I have already been vaccinated"):
        return "Yes"
    else:
        return vaccine_accept

def convertCountryToISO2(country_name):
    try:
        iso2_name = dict_countries[country_name]
        return iso2_name
    except:
        #print(country_name)
        return "XX"


def convertWaveToString(wave_id):
    try:
        if(wave_id!=""):
            wave_id = "wave" + str(int(float(wave_id)))
    except:
        wave_id = ""
    return wave_id


fn = os.path.join(os.path.dirname(__file__), '/data/covid_survey_responses.txt.gz')
data = pd.read_csv(fn, sep="\t", compression='gzip')
num_rows = data.shape[0]

col_names = ["id","start_date","country","age","gender","us_state","india_state",
             "vaccine_accept","effect_mask",'weight_demo',"future_vaccine","norms_vaccine",'wave']

data = data[col_names]
data = data.rename(columns={"id": "record_id"})

data = data.dropna(subset=["record_id","country","age","gender","wave"]) # these are required
data["wave"] = data["wave"].apply(convertWaveToString)

data["Vaccine Acceptance"] = data["vaccine_accept"]
data["vaccine_accept"] = data["vaccine_accept"].apply(mergeAlreadyVaccinated)

data.to_csv(os.path.join(os.path.dirname(__file__), "data/data_for_vaccine_analysis.txt"), sep="\t", index=False)