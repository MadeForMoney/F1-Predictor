import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

le_driver = joblib.load(os.path.join(BASE_DIR, "le_driver.pkl"))
le_track = joblib.load(os.path.join(BASE_DIR, "le_track.pkl"))
le_team = joblib.load(os.path.join(BASE_DIR, "le_team.pkl"))
