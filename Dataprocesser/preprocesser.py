import duckdb
import pandas as pd
import random
import numpy as np

class BatchDataPreprocesser:
    def __init__(self, tracking_path, play_path, players_path, batch_size=32, profile_path="data/出場選手プロフィール情報.xlsx"):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"CREATE TABLE tracking AS SELECT * FROM read_csv_auto('{tracking_path}')")
        self.con.execute(f"CREATE TABLE play AS SELECT * FROM read_csv_auto('{play_path}')")
        self.con.execute(f"CREATE TABLE players AS SELECT * FROM read_csv_auto('{players_path}')")
        self.batch_size = batch_size
        self.tracking_len = self.con.execute("SELECT COUNT(*) FROM tracking").fetchone()[0]
        
        # 攻撃数の取得
        self.attack_count = self.con.execute("SELECT COUNT(DISTINCT (攻撃履歴No, 試合ID)) FROM play WHERE 攻撃履歴No > 0").fetchone()[0]
        
        # playersテーブルから(試合ID, HA, 背番号)→選手IDの辞書を作成
        try:
            players_df = self.con.execute("SELECT * FROM players").df()
            
            # カラム名の標準化
            col_map = {}
            if '試合ID' in players_df.columns:
                col_map['試合ID'] = 'GameID'
            if 'ホームアウェイF' in players_df.columns:
                col_map['ホームアウェイF'] = 'HA'
            if '背番号' in players_df.columns:
                col_map['背番号'] = 'No'
            if '選手ID' in players_df.columns:
                col_map['選手ID'] = '選手ID'
            players_df = players_df.rename(columns=col_map)
            
            self.game_ha_no_to_playerid = {}
            for _, row in players_df.iterrows():
                try:
                    # データ型を統一してキーを作成
                    key = (int(row['GameID']), int(row['HA']), int(row['No']))
                    self.game_ha_no_to_playerid[key] = row['選手ID']
                except Exception as e:
                    print(f"[WARN] players.csv rowアクセス失敗: {e}, row={row}")
        except Exception as e:
            print(f"[WARN] playersテーブルの読み込みに失敗: {e}")
            self.game_ha_no_to_playerid = {}

    def __len__(self):
        return (self.attack_count + self.batch_size - 1) // self.batch_size

    def get_batch(self, idx=None):
        # 攻撃履歴Noと試合IDの組み合わせ一覧を取得
        attacks = self.con.execute("SELECT DISTINCT 攻撃履歴No, 試合ID FROM play WHERE 攻撃履歴No > 0 ORDER BY 試合ID, 攻撃履歴No").df()
        
        if idx is not None:
            # 指定されたバッチインデックスから攻撃を取得
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(attacks))
            batch_attacks = attacks.iloc[start_idx:end_idx]
        else:
            # ランダムサンプリング
            batch_attacks = attacks.sample(min(self.batch_size, len(attacks)))
        
        # 各攻撃のデータを取得
        batch = []
        for _, attack in batch_attacks.iterrows():
            attack_data = self.get_attack_batch(attack['攻撃履歴No'], attack['試合ID'])
            if attack_data is not None:
                batch.append(attack_data)
        
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
            print(f"[DEBUG] フレームデータなし: 試合{game_id} 攻撃{attack_no}")
            return None
        
        min_frame = play_rows['フレーム番号'].min()
        max_frame = play_rows['フレーム番号'].max()
        
        # trackingから該当範囲を抽出
        tracking_df = self.con.execute(f"SELECT * FROM tracking WHERE GameID={game_id} AND Frame >= {min_frame} AND Frame <= {max_frame} ORDER BY Frame").df()
        
        if tracking_df.empty:
            print(f"[DEBUG] トラッキングデータなし: 試合{game_id} 攻撃{attack_no} (フレーム範囲: {min_frame}-{max_frame})")
            return None
        
        # HA, Noごとに分割
        ha_no_groups = {}
        if not tracking_df.empty:
            for (ha, no), group in tracking_df.groupby(['HA', 'No']):
                ha_no_groups[(ha, no)] = group.reset_index(drop=True)
        
        if not ha_no_groups:
            print(f"[DEBUG] 選手データなし: 試合{game_id} 攻撃{attack_no}")
            return None
        
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

    def get_attack_tensor_batch(self, idx=None):
        """攻撃単位のテンソルバッチを取得"""
        attack_batch = self.get_batch(idx)
        
        tensors = []
        entities_list = []
        frames_list = []
        attack_info = []
        
        for attack_data in attack_batch:
            tensor, entities, frames = self.get_attack_tensor(
                attack_data['attack_no'], 
                attack_data['game_id']
            )
            tensors.append(tensor)
            entities_list.append(entities)
            frames_list.append(frames)
            attack_info.append({
                'attack_no': attack_data['attack_no'],
                'game_id': attack_data['game_id'],
                'min_frame': attack_data['min_frame'],
                'max_frame': attack_data['max_frame']
            })
        
        return {
            'tensors': tensors,
            'entities_list': entities_list,
            'frames_list': frames_list,
            'attack_info': attack_info
        }

    def get_attack_tensor(self, attack_no=None, game_id=None):
        batch = self.get_attack_batch(attack_no, game_id)
        
        # batchがNoneの場合はエラーを返す
        if batch is None:
            raise ValueError(f"攻撃データの取得に失敗: attack_no={attack_no}, game_id={game_id}")
        
        ha_no_groups = batch['ha_no_groups']
        frames = batch['tracking_frames']['Frame'].unique()
        frames.sort()
        
        # 選手交代情報を取得
        substitution_info = self.get_substitution_info(batch['game_id'], batch['attack_no'])
        
        # 実際に存在する選手番号を取得
        existing_players = list(ha_no_groups.keys())
        
        # ボールと選手を分離
        ball_key = (0, 0) if (0, 0) in ha_no_groups else None
        player_keys = [key for key in existing_players if key != (0, 0)]
        
        # 選手交代情報に基づいて選手を除外
        filtered_player_keys = self.filter_players_by_substitution(
            player_keys, substitution_info, batch['game_id']
        )
        
        # 最大22エンティティまで対応（ボール1 + 選手22）
        max_players = 22
        if len(filtered_player_keys) > max_players:
            print(f"[WARN] 選手数が多すぎます: {len(filtered_player_keys)} > {max_players}")
            # 最も多くのフレームに登場する選手を優先
            filtered_player_keys = self.select_most_frequent_players(
                filtered_player_keys, ha_no_groups, max_players
            )
        
        # entity_keysを動的に生成
        entity_keys = []
        if ball_key:
            entity_keys.append(ball_key)
        entity_keys.extend(filtered_player_keys)
        
        # 23エンティティに満たない場合はNoneで埋める
        while len(entity_keys) < 23:
            entity_keys.append(None)
        
        entity_id_map = {}
        
        # 現在の試合IDを取得
        current_game_id = batch['game_id']
        
        for i, (ha, no) in enumerate(entity_keys):
            if (ha, no) is None:
                entity_id_map[i] = None
            elif (ha, no) == (0, 0):
                entity_id_map[i] = 'ball'
            elif (ha, no) in ha_no_groups and not ha_no_groups[(ha, no)].empty:
                df = ha_no_groups[(ha, no)]
                game_id0 = df.iloc[0]['GameID']
                
                # データ型を統一してキーを作成
                key = (int(game_id0), int(ha), int(no))
                
                # players.csvから選手IDを特定
                pid = self.game_ha_no_to_playerid.get(key, None)
                if pid is None:
                    # 型を変えて再試行
                    alt_key = (game_id0, ha, no)
                    pid = self.game_ha_no_to_playerid.get(alt_key, None)
                
                entity_id_map[i] = pid
            else:
                entity_id_map[i] = None
        
        entities = [entity_id_map[i] for i in range(23)]
        tensor = np.full((23, len(frames), 2), np.nan, dtype=np.float32)
        frame_to_idx = {f: i for i, f in enumerate(frames)}
        
        for ent_idx, (ha, no) in enumerate(entity_keys):
            if (ha, no) is not None and (ha, no) in ha_no_groups:
                df = ha_no_groups[(ha, no)]
                for _, row in df.iterrows():
                    fidx = frame_to_idx[row['Frame']]
                    tensor[ent_idx, fidx, 0] = row['X']
                    tensor[ent_idx, fidx, 1] = row['Y']
        
        return tensor, entities, frames

    def get_substitution_info(self, game_id, attack_no):
        """選手交代情報を取得"""
        try:
            # この攻撃のフレーム範囲を取得
            play_rows = self.con.execute(f"""
                SELECT フレーム番号 FROM play 
                WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} 
                ORDER BY フレーム番号
            """).df()
            
            if play_rows.empty:
                return {}
            
            min_frame = play_rows['フレーム番号'].min()
            max_frame = play_rows['フレーム番号'].max()
            
            # 交代情報を取得（アクション名に「交代」が含まれるもの）
            substitutions = self.con.execute(f"""
                SELECT フレーム番号, 選手名, 選手背番号, ホームアウェイF, チームID
                FROM play 
                WHERE 試合ID={game_id} 
                AND フレーム番号 >= {min_frame} 
                AND フレーム番号 <= {max_frame}
                AND アクション名 LIKE '%交代%'
                ORDER BY フレーム番号
            """).df()
            
            substitution_info = {}
            for _, sub in substitutions.iterrows():
                frame = sub['フレーム番号']
                ha = sub['ホームアウェイF']
                no = sub['選手背番号']
                player_name = sub['選手名']
                team_id = sub['チームID']
                
                key = (ha, no)
                if key not in substitution_info:
                    substitution_info[key] = {
                        'player_name': player_name,
                        'team_id': team_id,
                        'substitution_frame': frame,
                        'is_substituted_out': True
                    }
            
            return substitution_info
            
        except Exception as e:
            print(f"[WARN] 交代情報の取得に失敗: {e}")
            return {}

    def filter_players_by_substitution(self, player_keys, substitution_info, game_id):
        """交代情報に基づいて選手を除外"""
        if not substitution_info:
            return player_keys
        
        # 交代された選手を除外
        filtered_keys = []
        for key in player_keys:
            if key not in substitution_info:
                filtered_keys.append(key)
            else:
                print(f"[INFO] 交代選手を除外: HA={key[0]}, No={key[1]}, 選手名={substitution_info[key]['player_name']}")
        
        return filtered_keys

    def select_most_frequent_players(self, player_keys, ha_no_groups, max_players):
        """最も多くのフレームに登場する選手を選択"""
        player_frames = {}
        
        for key in player_keys:
            if key in ha_no_groups:
                frame_count = len(ha_no_groups[key])
                player_frames[key] = frame_count
        
        # フレーム数でソート（降順）
        sorted_players = sorted(player_frames.items(), key=lambda x: x[1], reverse=True)
        
        # 上位max_playersを選択
        selected_keys = [key for key, _ in sorted_players[:max_players]]
        
        print(f"[INFO] 選手選択: {len(player_keys)} → {len(selected_keys)}")
        for key in selected_keys:
            print(f"  選択: HA={key[0]}, No={key[1]}, フレーム数={player_frames[key]}")
        
        return selected_keys

    def save_attack_data(self, save_dir="saved_attack_data", format="parquet"):
        """攻撃データをファイルとして保存"""
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.feather as feather
        
        # 保存ディレクトリを作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 全攻撃データを取得
        all_attacks = self.con.execute("""
            SELECT DISTINCT 試合ID, 攻撃履歴No 
            FROM play 
            WHERE 攻撃履歴No > 0
            ORDER BY 試合ID, 攻撃履歴No
        """).df()
        
        print(f"保存開始: {len(all_attacks)}個の攻撃データ (形式: {format})")
        
        for idx, (_, attack) in enumerate(all_attacks.iterrows()):
            try:
                # 攻撃データを取得
                attack_data = self.get_attack_batch(attack['攻撃履歴No'], attack['試合ID'])
                if attack_data is None:
                    print(f"[WARN] 攻撃データが取得できません: {attack['試合ID']}_{attack['攻撃履歴No']} (フレームデータなし)")
                    continue
                
                # テンソルデータを取得
                try:
                    tensor, entities, frames = self.get_attack_tensor(
                        attack['攻撃履歴No'], 
                        attack['試合ID']
                    )
                except ValueError as e:
                    print(f"[WARN] 攻撃データの取得に失敗: {attack['試合ID']}_{attack['攻撃履歴No']} - {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] テンソル生成に失敗: {attack['試合ID']}_{attack['攻撃履歴No']} - {e}")
                    continue
                
                # ファイル名を生成
                filename = f"attack_{attack['試合ID']}_{attack['攻撃履歴No']:04d}"
                
                # テンソルデータをDataFrameに変換
                tensor_df = pd.DataFrame({
                    'entity_idx': np.repeat(range(tensor.shape[0]), tensor.shape[1]),
                    'frame_idx': np.tile(range(tensor.shape[1]), tensor.shape[0]),
                    'x': tensor.reshape(-1, 2)[:, 0],
                    'y': tensor.reshape(-1, 2)[:, 1]
                })
                
                # メタデータをDataFrameに変換
                meta_df = pd.DataFrame({
                    'attack_no': [attack['攻撃履歴No']],
                    'game_id': [attack['試合ID']],
                    'min_frame': [attack_data['min_frame']],
                    'max_frame': [attack_data['max_frame']],
                    'frame_count': [len(frames)],
                    'tensor_shape_0': [tensor.shape[0]],
                    'tensor_shape_1': [tensor.shape[1]],
                    'tensor_shape_2': [tensor.shape[2]],
                    'entities_count': [len(entities)]
                })
                
                # entitiesとframesを文字列として保存
                meta_df['entities'] = [str(entities)]
                meta_df['frames'] = [str(frames.tolist())]
                
                # 形式に応じて保存
                if format.lower() == "parquet":
                    # Parquet形式で保存
                    tensor_path = os.path.join(save_dir, f"{filename}_tensor.parquet")
                    meta_path = os.path.join(save_dir, f"{filename}_meta.parquet")
                    
                    tensor_df.to_parquet(tensor_path, index=False)
                    meta_df.to_parquet(meta_path, index=False)
                    
                elif format.lower() == "feather":
                    # Feather形式で保存
                    tensor_path = os.path.join(save_dir, f"{filename}_tensor.feather")
                    meta_path = os.path.join(save_dir, f"{filename}_meta.feather")
                    
                    tensor_df.to_feather(tensor_path)
                    meta_df.to_feather(meta_path)
                
                if (idx + 1) % 100 == 0:
                    print(f"保存進捗: {idx + 1}/{len(all_attacks)}")
                    
            except Exception as e:
                print(f"エラー: 攻撃{attack['試合ID']}_{attack['攻撃履歴No']}の保存に失敗: {e}")
                continue
        
        print(f"保存完了: {save_dir}")

    def load_attack_data(self, filename, data_dir="saved_attack_data", format="parquet"):
        """保存された攻撃データを読み込み"""
        import os
        import ast
        
        # ファイル名から基本名を取得
        base_name = filename.replace('_tensor.parquet', '').replace('_meta.parquet', '')
        base_name = base_name.replace('_tensor.feather', '').replace('_meta.feather', '')
        
        # ファイルパスを構築
        if format.lower() == "parquet":
            tensor_path = os.path.join(data_dir, f"{base_name}_tensor.parquet")
            meta_path = os.path.join(data_dir, f"{base_name}_meta.parquet")
        elif format.lower() == "feather":
            tensor_path = os.path.join(data_dir, f"{base_name}_tensor.feather")
            meta_path = os.path.join(data_dir, f"{base_name}_meta.feather")
        
        if not os.path.exists(tensor_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {tensor_path} または {meta_path}")
        
        # データを読み込み
        if format.lower() == "parquet":
            tensor_df = pd.read_parquet(tensor_path)
            meta_df = pd.read_parquet(meta_path)
        elif format.lower() == "feather":
            tensor_df = pd.read_feather(tensor_path)
            meta_df = pd.read_feather(meta_path)
        
        # テンソルを再構築
        shape = (meta_df['tensor_shape_0'].iloc[0], 
                meta_df['tensor_shape_1'].iloc[0], 
                meta_df['tensor_shape_2'].iloc[0])
        
        tensor = np.full(shape, np.nan, dtype=np.float32)
        for _, row in tensor_df.iterrows():
            entity_idx = int(row['entity_idx'])
            frame_idx = int(row['frame_idx'])
            tensor[entity_idx, frame_idx, 0] = row['x']
            tensor[entity_idx, frame_idx, 1] = row['y']
        
        # entitiesとframesを復元
        entities = ast.literal_eval(meta_df['entities'].iloc[0])
        frames = np.array(ast.literal_eval(meta_df['frames'].iloc[0]))
        
        # データを辞書形式で返す
        data = {
            'tensor': tensor,
            'entities': entities,
            'frames': frames,
            'info': {
                'attack_no': meta_df['attack_no'].iloc[0],
                'game_id': meta_df['game_id'].iloc[0],
                'min_frame': meta_df['min_frame'].iloc[0],
                'max_frame': meta_df['max_frame'].iloc[0],
                'frame_count': meta_df['frame_count'].iloc[0]
            }
        }
        
        return data

    def get_attack_data_by_info(self, game_id, attack_no, data_dir="saved_attack_data", format="parquet"):
        """試合IDと攻撃履歴Noからデータを取得"""
        filename = f"attack_{game_id}_{attack_no:04d}_tensor.{format}"
        return self.load_attack_data(filename, data_dir, format)

    def list_saved_attacks(self, data_dir="saved_attack_data", format="parquet"):
        """保存された攻撃データの一覧を取得"""
        import os
        
        if not os.path.exists(data_dir):
            return []
        
        attacks = []
        for filename in os.listdir(data_dir):
            if filename.endswith(f'_meta.{format}'):
                # ファイル名から情報を抽出
                parts = filename.replace(f'_meta.{format}', '').split('_')
                if len(parts) >= 3:
                    game_id = parts[1]
                    attack_no = int(parts[2])
                    
                    # メタデータを読み込み
                    meta_path = os.path.join(data_dir, filename)
                    if format == "parquet":
                        meta_df = pd.read_parquet(meta_path)
                    else:
                        meta_df = pd.read_feather(meta_path)
                    
                    meta_data = {
                        'filename': filename,
                        'attack_no': attack_no,
                        'game_id': game_id,
                        'min_frame': meta_df['min_frame'].iloc[0],
                        'max_frame': meta_df['max_frame'].iloc[0],
                        'frame_count': meta_df['frame_count'].iloc[0],
                        'tensor_shape': (meta_df['tensor_shape_0'].iloc[0], 
                                       meta_df['tensor_shape_1'].iloc[0], 
                                       meta_df['tensor_shape_2'].iloc[0]),
                        'entities_count': meta_df['entities_count'].iloc[0]
                    }
                    attacks.append(meta_data)
        
        return sorted(attacks, key=lambda x: (x['game_id'], x['attack_no']))

    def get_random_attack_data(self, data_dir="saved_attack_data", format="parquet"):
        """ランダムな攻撃データを取得"""
        import os
        import random
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"データディレクトリが見つかりません: {data_dir}")
        
        # メタファイルの一覧を取得
        meta_files = [f for f in os.listdir(data_dir) if f.endswith(f'_meta.{format}')]
        if not meta_files:
            raise FileNotFoundError("保存されたデータが見つかりません")
        
        # ランダムにファイルを選択
        random_file = random.choice(meta_files)
        return self.load_attack_data(random_file, data_dir, format)

    def get_batch_from_saved_data(self, batch_size=5, data_dir="saved_attack_data", format="parquet"):
        """保存されたデータからバッチを取得"""
        import os
        import random
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"データディレクトリが見つかりません: {data_dir}")
        
        # メタファイルの一覧を取得
        meta_files = [f for f in os.listdir(data_dir) if f.endswith(f'_meta.{format}')]
        if not meta_files:
            raise FileNotFoundError("保存されたデータが見つかりません")
        
        # ランダムにファイルを選択
        selected_files = random.sample(meta_files, min(batch_size, len(meta_files)))
        
        batch_data = []
        for filename in selected_files:
            data = self.load_attack_data(filename, data_dir, format)
            batch_data.append(data)
        
        return batch_data

    def close(self):
        self.con.close()

if __name__ == "__main__":
    loader = BatchDataPreprocesser(
        tracking_path="data/2022_data/*/tracking.csv",
        play_path="data/2022_data/*/play.csv",
        players_path="data/2022_data/*/players.csv",
        batch_size=5  # 1バッチ = 5攻撃
    )

    print(f"\n=== 攻撃単位バッチ処理 ===")
    print(f"総攻撃数: {loader.attack_count}")
    print(f"バッチ数: {len(loader)}")
    print(f"1バッチあたりの攻撃数: {loader.batch_size}")
    
    # 最初のバッチを取得
    batch_data = loader.get_attack_tensor_batch(0)
    print(f"\n最初のバッチの攻撃数: {len(batch_data['tensors'])}")
    
    for i, (tensor, entities, frames, info) in enumerate(zip(
        batch_data['tensors'], 
        batch_data['entities_list'], 
        batch_data['frames_list'], 
        batch_data['attack_info']
    )):
        print(f"\n攻撃 {i+1}:")
        print(f"  攻撃履歴No: {info['attack_no']}, 試合ID: {info['game_id']}")
        print(f"  フレーム範囲: {info['min_frame']} ～ {info['max_frame']}")
        print(f"  テンソルshape: {tensor.shape}")
        print(f"  フレーム数: {len(frames)}")
        print(f"  選手ID: {entities}")
    
    # データ保存のテスト
    print(f"\n=== データ保存テスト (Parquet) ===")
    loader.save_attack_data("test_saved_data_parquet", format="parquet")
    
    print(f"\n=== データ保存テスト (Feather) ===")
    loader.save_attack_data("test_saved_data_feather", format="feather")
    
    # 保存されたデータの一覧を取得
    print(f"\n=== 保存されたデータ一覧 (Parquet) ===")
    saved_attacks_parquet = loader.list_saved_attacks("test_saved_data_parquet", format="parquet")
    print(f"保存された攻撃数: {len(saved_attacks_parquet)}")
    for attack in saved_attacks_parquet[:3]:  # 最初の3件を表示
        print(f"  {attack['filename']}: 試合{attack['game_id']} 攻撃{attack['attack_no']}")
    
    print(f"\n=== 保存されたデータ一覧 (Feather) ===")
    saved_attacks_feather = loader.list_saved_attacks("test_saved_data_feather", format="feather")
    print(f"保存された攻撃数: {len(saved_attacks_feather)}")
    for attack in saved_attacks_feather[:3]:  # 最初の3件を表示
        print(f"  {attack['filename']}: 試合{attack['game_id']} 攻撃{attack['attack_no']}")
    
    # 特定の攻撃データを読み込み (Parquet)
    print(f"\n=== 特定攻撃データの読み込みテスト (Parquet) ===")
    if saved_attacks_parquet:
        test_attack = saved_attacks_parquet[0]
        loaded_data = loader.get_attack_data_by_info(
            test_attack['game_id'], 
            test_attack['attack_no'], 
            "test_saved_data_parquet",
            format="parquet"
        )
        print(f"読み込み成功: {test_attack['filename']}")
        print(f"  テンソルshape: {loaded_data['tensor'].shape}")
        print(f"  選手ID: {loaded_data['entities']}")
        print(f"  フレーム数: {len(loaded_data['frames'])}")
        print(f"  情報: {loaded_data['info']}")
    
    # 特定の攻撃データを読み込み (Feather)
    print(f"\n=== 特定攻撃データの読み込みテスト (Feather) ===")
    if saved_attacks_feather:
        test_attack = saved_attacks_feather[0]
        loaded_data = loader.get_attack_data_by_info(
            test_attack['game_id'], 
            test_attack['attack_no'], 
            "test_saved_data_feather",
            format="feather"
        )
        print(f"読み込み成功: {test_attack['filename']}")
        print(f"  テンソルshape: {loaded_data['tensor'].shape}")
        print(f"  選手ID: {loaded_data['entities']}")
        print(f"  フレーム数: {len(loaded_data['frames'])}")
        print(f"  情報: {loaded_data['info']}")
    
    # ランダムな攻撃データを読み込み (Parquet)
    print(f"\n=== ランダム攻撃データの読み込みテスト (Parquet) ===")
    random_data = loader.get_random_attack_data("test_saved_data_parquet", format="parquet")
    print(f"ランダム読み込み成功")
    print(f"  テンソルshape: {random_data['tensor'].shape}")
    print(f"  選手ID: {random_data['entities']}")
    print(f"  フレーム数: {len(random_data['frames'])}")
    print(f"  情報: {random_data['info']}")
    
    # 保存されたデータからバッチを取得 (Parquet)
    print(f"\n=== 保存データからのバッチ取得テスト (Parquet) ===")
    saved_batch = loader.get_batch_from_saved_data(3, "test_saved_data_parquet", format="parquet")
    print(f"バッチ取得成功: {len(saved_batch)}個の攻撃")
    for i, data in enumerate(saved_batch):
        print(f"  攻撃{i+1}: 試合{data['info']['game_id']} 攻撃{data['info']['attack_no']}")
    
    # 保存されたデータからバッチを取得 (Feather)
    print(f"\n=== 保存データからのバッチ取得テスト (Feather) ===")
    saved_batch = loader.get_batch_from_saved_data(3, "test_saved_data_feather", format="feather")
    print(f"バッチ取得成功: {len(saved_batch)}個の攻撃")
    for i, data in enumerate(saved_batch):
        print(f"  攻撃{i+1}: 試合{data['info']['game_id']} 攻撃{data['info']['attack_no']}")
    
    loader.close()