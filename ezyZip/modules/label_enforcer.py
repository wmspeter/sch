"""
label_enforcer.py - enforce mapping to fixed 35 groups and 20 emotions.
"""
from difflib import get_close_matches
GROUPS = [
"Academics","Teachers","Classmates","Homework","Exams & Tests","Learning Motivation","School Environment",
"Classroom Facilities","School Cleanliness","Library","Canteen / Food","Health (Physical)",
"Mental Health / Psychology","Sports / Physical Education","Extracurricular Activities","School Events",
"Technology / Devices / Online Learning","Transportation / School Bus","Safety / Bullying","School Management / Rules",
"Family / Home","Parents","Future / Career Orientation","Finance / Allowance","Friends / Relationships","Love / Crush",
"Sleep / Fatigue","Stress / Pressure","Appearance / Body Image","Teachers’ Attitude / Behavior","Peer Pressure",
"School Schedule / Time Table","Homework Load","Holidays / Vacations","Others"
]
EMOTIONS = [
"Happy","Sad","Angry","Anxious","Excited","Bored","Neutral","Proud","Embarrassed","Lonely",
"Relieved","Frustrated","Hopeful","Fearful","Disappointed","Grateful","Curious","Confused","Tired","Annoyed"
]
def enforce_group_label(suggested: str) -> str:
    if suggested in GROUPS:
        return suggested
    matches = get_close_matches(suggested, GROUPS, n=1, cutoff=0.6)
    return matches[0] if matches else "Others"
def enforce_emotion_label(suggested: str) -> str:
    if suggested in EMOTIONS:
        return suggested
    matches = get_close_matches(suggested, EMOTIONS, n=1, cutoff=0.5)
    return matches[0] if matches else "Neutral"
