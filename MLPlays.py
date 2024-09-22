import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
today = dt.date.today()
year = today.year

df = nfl.import_schedules(years=range(year-5,year+1))
currSeason = df[df.season == year]
predWeek = currSeason[['week', 'total']].dropna()
if np.isnan(predWeek.week.max()):
    predWeek = 1
else:
    predWeek = predWeek.week.max() + 1

#predWeek=3

# Prepare dataframe by dropping irrelevant predictors and formatting columns for KNN
df = df[df.result != 0]
df['Home'] = np.where(df['result'] > 0, 1, 0)

def date_to_month(time_str):
    year, month, day = map(int, time_str.split('-'))
    return month
df['month'] = df['gameday'].apply(date_to_month)
# Function to convert time to seconds
def time_to_seconds(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60
# Apply the function to the 'time' column
df['gametime'] = df['gametime'].apply(time_to_seconds)

dict_day = {"weekday": {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}}
df.replace(dict_day, inplace=True)
dict_roof = {"roof": {"outdoors": 0, "dome": 1, "closed": 2, "open": 3}}
df.replace(dict_roof, inplace=True)
dict_surface = {"surface": {"grass": 0, "grass ": 0, "fieldturf": 1, "astroturf": 2, "sportturf": 3, "matrixturf": 4, "astroplay": 5, "a_turf": 6, "dessograss": 7}}
df.replace(dict_surface, inplace=True)

df_dummy = pd.get_dummies(df, drop_first=True, columns=['game_type', 'location', 'stadium_id', 'home_team', 'away_team', 'home_qb_id', 'away_qb_id', 'home_coach', 'away_coach'])
features = df_dummy.drop(['Home', 'home_moneyline', 'away_moneyline', 'gameday', 'surface', 'game_id', 'home_score', 'away_score', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'away_qb_name', 'home_qb_name', 'referee', 'stadium', 'wind', 'temp'], axis=1).columns

train_df = df_dummy[(df_dummy.season < year) | ((df_dummy.season == year) & (df_dummy.week < predWeek))]
test_df = df_dummy[(df_dummy.season == year) & (df_dummy.week == predWeek)]
train_df.dropna(inplace=True)
X_train = train_df[features]
y_train = train_df.Home
X_test = test_df[features]
y_test = test_df.Home

model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=6,  # Limit the depth of the trees
    learning_rate=0.1,
    n_estimators=1000,  # Use a large number of trees
    reg_alpha=0.1,  # L1 regularization term on weights
    reg_lambda=0.1,  # L2 regularization term on weights
    early_stopping_rounds=10  # Stop early if validation score doesn't improve
)

# Evaluation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predict probabilities and classes on selected features
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Append predictions and probabilities to X_test
X_test['Prediction'] = y_pred
X_test['Home Probability'] = y_pred_proba

# Predicted Plays log
nextPlays = pd.merge(right=X_test, left=currSeason, right_index=True, left_index=True, how='left')
nextPlays = nextPlays[nextPlays.Prediction == 1]
nextPlays = nextPlays[['game_id', 'season_x', 'week_x', 'home_team', 'away_team', 'gametime_x', 'weekday_x', 'spread_line_x', 'home_moneyline', 'Home Probability']]
nextPlays.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Spread Line', 'Home Moneyline', 'Home Probability']
nextPlays = nextPlays[nextPlays.Week == predWeek]
nextPlays['Home Implied Odds'] = np.where(nextPlays['Home Moneyline'] < 0, (abs(nextPlays['Home Moneyline'])/(abs(nextPlays['Home Moneyline'])+100)), (100/(nextPlays['Home Moneyline']+100)))
# Value cleanup
dict_day = {"Day": {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}}
nextPlays.replace(dict_day, inplace=True)

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("NFL Matchup Predictor")

# Add weekly plays
worksheet1 = sh.worksheet("Plays")
worksheet1.append_rows(nextPlays.values.tolist())