import math
import datetime
from copy import deepcopy

def get_user_data():
    return {
        "name": ask("What is your name?"),
        "birth_year": int(ask("In which year were you born?")),
        "birth_month": int(ask("In which month were you born?")),
        "birth_day": int(ask("On which day of the month were you born?")),
    }

def ask(question):
    print(question)
    result = input()
    return result

def get_age_in_days(user_data):
    birthdate = datetime.datetime(user_data["birth_year"], user_data["birth_month"], user_data["birth_day"])
    return (datetime.datetime.now() - birthdate).days

def calculate_biorythm(age):
    return { 
        "physical": math.sin(2*math.pi*age/23), 
        "emotional": math.sin(2*math.pi*age/28),
        "intellectual": math.sin(2*math.pi*age/33)
    }

def get_predictions(biorythm):
    total = sum(biorythm.values())
    return total / len(biorythm)

def write_predictions(prediction):
    if prediction > 0.5:
        print("Your prediction is: High")
    elif prediction > -0.5:
        print("Your prediction is: Normal")
    else:
        print("Your prediction is: Bad")

def main():
    # user_data = get_user_data()
    user_data = {"birth_year": 2001, "birth_month": 12, "birth_day": 12}
    age = get_age_in_days(user_data)
    biorythm = calculate_biorythm(age)
    prediction = get_predictions(biorythm)
    write_predictions(prediction)
    
    # Calculate and display predictions for tomorrow
    biorythm_tomorrow = calculate_biorythm(age + 1)
    prediction_tomorrow = get_predictions(biorythm_tomorrow)
    write_predictions(prediction_tomorrow)

main()