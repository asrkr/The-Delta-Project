import os
from src.data_manager import update_database, update_latest_qualifying, update_sprint_data, update_calendar, extract_fastf1_features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

if __name__ == "__main__":
    print("1. Update Ergast results")
    print("2. Update Ergast results (single season)")
    print("3. Update race calendar")
    print("4. Update FastF1 extra features")
    print("5. Update FastF1 extra features (single season)")
    print("6. Update the latest grid")
    print("7. Update sprint results (Ergast)")
    print("8. Update sprint results (single season)")
    print("9. Update ALL")
    print("10) Exit")

    choice = int(input("Choice: "))

    if choice == 1:
        print("Ergast results update.")
        y1 = int(input("Start year (recommended: 2001): "))
        y2 = int(input("End year (latest season): "))
        update_database(y1, y2)


    elif choice == 2:
        print("Ergast results update.")
        y = int(input("Season to update (from 2001 to latest season): "))
        update_database(y, y)

    
    elif choice == 3:
        print("Calendar update.")
        y1 = int(input("Start year (recommended: 2001): "))
        y2 = int(input("End year (latest season): "))
        update_calendar(y1, y2)
    
    
    elif choice == 4:
        print("FastF1 features update.")
        y1 = int(input("Start year (not prior than 2018): "))
        y2 = int(input("End year (latest season): "))
        extract_fastf1_features(y1, y2)


    elif choice == 5:
        print("FastF1 features update.")
        y = int(input("Season to update (no prior than 2018): "))
        extract_fastf1_features(y, y)
    

    elif choice == 6:
        y = int(input(("Year: ")))
        rnd = int(input("Round: "))
        update_latest_qualifying(y, rnd)
    

    elif choice == 7:
        print("Sprint results update.")
        y1 = int(input("Start year (sprints started in 2021): "))
        y2 = int(input("End year (latest season): "))
        update_sprint_data(y1, y2)
    

    elif choice == 8:
        print("Sprint results update.")
        y = int(input("Season to update (sprints started in 2021): "))
        update_sprint_data(y, y)


    elif choice == 9:
        print("Full update.")
        y1 = int(input("Start year for Ergast (recommended: 2001): "))
        y2 = int(input("Start year for FastF1 (not prior than 2018): "))
        y3 = int(input("End year (latest season): "))
        rnd = int(input("Latest round (for qualifyings): "))
        update_database(y1, y3)
        update_calendar(y1, y3)
        extract_fastf1_features(y2, y3)
        update_latest_qualifying(y3, rnd)
    

    elif choice == 10:
        exit()

    
    else:
        print("Invalid choice")
