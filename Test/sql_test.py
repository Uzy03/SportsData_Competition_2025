import sys
import os
import random
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../Dataprocesser'))
from spl_base import load_to_database

# tracking.csvのパス
tracking_path = "data/2022_data/*/tracking.csv"

# play.csvのパス
play_path = "data/2022_data/*/play.csv"

# players.csvのパス
players_path = "data/2022_data/*/players.csv"

# データベースにロード
con = load_to_database(tracking_path, play_path, players_path)
assert con is not None, "DBロード失敗"

"""
# SQLクエリ例: テーブルのカラム名を取得
try:
    df_tracking = con.execute("PRAGMA table_info(tracking)").df()
    print("カラム情報:")
    print(df_tracking)
except Exception as e:
    print(f"クエリエラー: {e}")

try:
    df_play = con.execute("PRAGMA table_info(play)").df()
    print("カラム情報:")
    print(df_play)
except Exception as e:
    print(f"クエリエラー: {e}")

try:
    df_players = con.execute("PRAGMA table_info(players)").df()
    print("カラム情報:")
    print(df_players)
except Exception as e:
    print(f"クエリエラー: {e}")

# SQLクエリ例: レコード数を取得
try:
    df_tracking = con.execute("SELECT COUNT(*) as cnt FROM tracking").df()
    print("レコード数:")
    print(df_tracking)
except Exception as e:
    print(f"クエリエラー: {e}")

try:
    df_play = con.execute("SELECT COUNT(*) as cnt FROM play").df()
    print("レコード数:")
    print(df_play)
except Exception as e:
    print(f"クエリエラー: {e}")

try:
    df_players = con.execute("SELECT COUNT(*) as cnt FROM players").df()
    print("レコード数:")
    print(df_players)
except Exception as e:
    print(f"クエリエラー: {e}")
"""

# SQLクエリ例: 選手の情報を取得

# trackingからランダムに1行取得
tracking_df = con.execute("SELECT * FROM tracking").df()
random_row = tracking_df.sample(1).iloc[0]
game_id = random_row['GameID']
ha = random_row['HA']
no = random_row['No']
print(f"ランダム選択: GameID={game_id}, HA={ha}, No={no}")

# playから該当選手名を特定
play_query = f"SELECT 選手名 FROM play WHERE 試合ID={game_id} AND ホームアウェイF={ha} AND 選手背番号={no} LIMIT 1"
play_df = con.execute(play_query).df()
if play_df.empty:
    print("playに該当選手なし")
else:
    player_name = play_df.iloc[0]['選手名']
    print(f"該当選手名: {player_name}")
    # playersからその選手名の情報を取得
    players_query = f"SELECT * FROM players WHERE 試合ID={game_id} AND ホームアウェイF={ha} AND 背番号={no} AND 選手名='{player_name}'"
    players_df = con.execute(players_query).df()
    if players_df.empty:
        print("playersに該当選手情報なし")
    else:
        print("players情報:")
        print(players_df)

con.close()
