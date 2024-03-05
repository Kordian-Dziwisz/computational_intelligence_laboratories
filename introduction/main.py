import math
import datetime
from copy import copy


# I've spent about 45 minutes writing this
# AI wrote something similar in 5 minutes, with explaining, that's scary 

def get_user_data():
    return {
        "name": ask("jak masz na imię?"),
        "birthyear": int(ask("którego roku się urodziłeś?")),
        "birthmonth": int(ask("którego miesiąca się urodziłeś?")),
        "birthday": int(ask("którego dnia miesiąca się urodziłeś?")),
    }


def ask(question):
    print(question)
    return input()


def get_age_in_days(user_data):
    return (
        datetime.datetime.now()
        - datetime.datetime(
            user_data.get("birthyear"),
            user_data.get("birthmonth"),
            user_data.get("birthday"),
        )
    ).days


def get_biorythm(age):
    return { "physical": math.sin(2*math.pi*age/23), 
    "emotional": math.sin(2*math.pi*age/28),
    "intelectual": math.sin(2*math.pi*age/33)
    }

def get_predictions(biorythm):
    biorythm = copy(biorythm)
    sum = 0
    for key, value in biorythm.items():
        sum += value
    return sum/len(biorythm)

def write_predictions(biorythm):
    prediciton = get_predictions(biorythm)
    if(prediciton>0.5):
        print("Your predictions is: high")
    elif(prediciton>-0.5):
        print("Your predictions is: normal")
    else:
        print("Your prediciton is: bad")

def main():
    # result = get_user_data()
    # print(result)
    user_data = {"birthyear": 2001, "birthmonth": 12, "birthday": 12}
    biorythm = get_biorythm(get_age_in_days(user_data))
    write_predictions(biorythm)
    biorythm_tomorrow = get_biorythm(get_age_in_days(user_data)+1)
    write_predictions(biorythm_tomorrow)


main()
