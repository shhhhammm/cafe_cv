from settings import *
from datetime import datetime

people_amounts = []  # a list of number of people, its average is a number of people in the room,
                     # refreshes every <settings.TIMER> minutes

def average():
    global people_amounts
    return sum(people_amounts) / len(people_amounts) if people_amounts != [] else 0


def refresh(delta_time):
    global people_amounts
    if delta_time >= TIMER:
        if SAVE_DATA:
            output_file = open("output.txt", 'a')
            output_file.write(f"{datetime.now()} : {average()}\n")
            output_file.close()

        people_amounts = []
