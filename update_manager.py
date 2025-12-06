import os
from src.data_manager import update_database, update_calendar, extract_fastf1_features, update_latest_qualifying

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

if __name__ == "__main__":
    print("1. Update Ergast results")
    print("2. Update Ergast results (only one season): ")
    print("3. Update race calendar")
    print("4. Update FastF1 extra features")
    print("5. Update FastF1 extra features (only one season): ")
    print("6. Update the latest grid: ")
    print("7. Update ALL")
    print("8) Exit")

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
        print("Full update.")
        y1 = int(input("Start year for Ergast (recommended: 2001): "))
        y2 = int(input("Start year for FastF1 (not prior than 2018): "))
        y3 = int(input("End year (latest season): "))
        update_database(y1, y3)
        update_calendar(y1, y3)
        extract_fastf1_features(y2, y3)
    

    elif choice == 8:
        exit()

    
    else:
        print("Invalid choice")
