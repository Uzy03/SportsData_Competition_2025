import duckdb
import pandas as pd
import random

class SoccerBatchDataLoader:
    def __init__(self, tracking_path, play_path, players_path, batch_size=32):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"CREATE TABLE tracking AS SELECT * FROM read_csv_auto('{tracking_path}')")
        self.con.execute(f"CREATE TABLE play AS SELECT * FROM read_csv_auto('{play_path}')")
        self.con.execute(f"CREATE TABLE players AS SELECT * FROM read_csv_auto('{players_path}')")
        self.batch_size = batch_size
        self.tracking_len = self.con.execute("SELECT COUNT(*) FROM tracking").fetchone()[0]

    def __len__(self):
        return (self.tracking_len + self.batch_size - 1) // self.batch_size

    def get_batch(self, idx=None):
        # idxが指定されていればそのバッチ、なければランダムサンプリング
        if idx is not None:
            offset = idx * self.batch_size
            query = f"SELECT * FROM tracking LIMIT {self.batch_size} OFFSET {offset}"
            batch_df = self.con.execute(query).df()
        else:
            # ランダムサンプリング
            indices = random.sample(range(self.tracking_len), self.batch_size)
            indices_str = ",".join(map(str, indices))
            batch_df = self.con.execute(f"SELECT * FROM tracking WHERE rowid IN ({indices_str})").df()
        # それぞれの行についてplay, players情報を紐付けて返す
        batch = []
        for _, row in batch_df.iterrows():
            game_id = row['GameID']
            ha = row['HA']
            no = row['No']
            # playから選手名
            play_df = self.con.execute(f"SELECT 選手名 FROM play WHERE 試合ID={game_id} AND ホームアウェイF={ha} AND 選手背番号={no} LIMIT 1").df()
            player_name = play_df.iloc[0]['選手名'] if not play_df.empty else None
            # playersから情報
            if player_name:
                players_df = self.con.execute(f"SELECT * FROM players WHERE 試合ID={game_id} AND ホームアウェイF={ha} AND 背番号={no} AND 選手名='{player_name}'").df()
            else:
                players_df = pd.DataFrame()
            batch.append({
                'tracking': row,
                'player_name': player_name,
                'players_info': players_df
            })
        return batch

    def get_attack_batch(self, attack_no=None, game_id=None):
        # 攻撃履歴Noと試合IDの組み合わせ一覧を取得
        attacks = self.con.execute("SELECT DISTINCT 攻撃履歴No, 試合ID FROM play WHERE 攻撃履歴No > 0").df()
        if attack_no is None or game_id is None:
            # ランダムに1つ選ぶ
            attack = attacks.sample(1).iloc[0]
            attack_no = attack['攻撃履歴No']
            game_id = attack['試合ID']
        # この攻撃のフレーム範囲を特定
        play_rows = self.con.execute(f"SELECT フレーム番号 FROM play WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} ORDER BY フレーム番号").df()
        if play_rows.empty:
            return None
        min_frame = play_rows['フレーム番号'].min()
        max_frame = play_rows['フレーム番号'].max()
        # trackingから該当範囲を抽出
        tracking_df = self.con.execute(f"SELECT * FROM tracking WHERE GameID={game_id} AND Frame >= {min_frame} AND Frame <= {max_frame} ORDER BY Frame").df()
        # HA, Noごとに分割
        ha_no_groups = {}
        if not tracking_df.empty:
            for (ha, no), group in tracking_df.groupby(['HA', 'No']):
                ha_no_groups[(ha, no)] = group.reset_index(drop=True)
        # play, players情報は必要に応じて個別に参照できるように返す
        return {
            'attack_no': attack_no,
            'game_id': game_id,
            'min_frame': min_frame,
            'max_frame': max_frame,
            'tracking_frames': tracking_df,
            'ha_no_groups': ha_no_groups,  # 追加: (HA, No)ごとのDataFrame辞書
            'play_rows': self.con.execute(f"SELECT * FROM play WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} ORDER BY フレーム番号").df()
        }

    def close(self):
        self.con.close()

if __name__ == "__main__":
    loader = SoccerBatchDataLoader(
        tracking_path="data/2022_data/*/tracking.csv",
        play_path="data/2022_data/*/play.csv",
        players_path="data/2022_data/*/players.csv",
        batch_size=32
    )
    """
    print("=== 通常バッチ ===")
    batch = loader.get_batch()
    i = 0
    for sample in batch:
        i += 1
        print(f"\nbatch[{i}]")
        print(sample['tracking'])
        print("Player Name:", sample['player_name'])
        print("Players Info:")
        print(sample['players_info'])
    """

    print("\n=== 攻撃単位バッチ ===")
    attack_batch = loader.get_attack_batch()
    print(f"攻撃履歴No: {attack_batch['attack_no']}, 試合ID: {attack_batch['game_id']}")
    print(f"フレーム範囲: {attack_batch['min_frame']} ～ {attack_batch['max_frame']}")
    print(f"trackingフレーム数: {len(attack_batch['tracking_frames'])}")
    print("(HA, No)ごとのデータフレーム数:", len(attack_batch['ha_no_groups']))
    for key, df in attack_batch['ha_no_groups'].items():
        print(f"  HA={key[0]}, No={key[1]}: {(df)}行")
    #print(attack_batch['tracking_frames'].head())
    loader.close()