import sys
import duckdb


def load_to_database(tracking_path="tracking.csv", play_path="play.csv", players_path="players.csv"):
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"CREATE TABLE tracking AS SELECT * FROM read_csv_auto('{tracking_path}')")
    except Exception as e:
        print(f"Failed to load {tracking_path}: {e}")
        return None
    try:
        con.execute(f"CREATE TABLE play AS SELECT * FROM read_csv_auto('{play_path}')")
    except Exception as e:        
        print(f"Failed to load {play_path}: {e}")
        return None
    try:
        con.execute(f"CREATE TABLE players AS SELECT * FROM read_csv_auto('{players_path}')")
    except Exception as e:
        print(f"Failed to load {players_path}: {e}")
        return None
    return con