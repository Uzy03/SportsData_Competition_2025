import duckdb
import pandas as pd
import random
import numpy as np

class BatchDataLoader:
    """
    サッカーのトラッキングデータを攻撃シーン単位で処理し、
    Parquet/Feather形式で保存するためのクラス
    """
    
    def __init__(self, tracking_path, play_path, players_path, profile_path="data/出場選手プロフィール情報.xlsx"):
        """
        初期化
        
        Args:
            tracking_path: トラッキングデータのパス
            play_path: プレイデータのパス
            players_path: 選手データのパス
            profile_path: プロフィールデータのパス（未使用）
        """
        # DuckDBデータベースの初期化
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"CREATE TABLE tracking AS SELECT * FROM read_csv_auto('{tracking_path}')")
        self.con.execute(f"CREATE TABLE play AS SELECT * FROM read_csv_auto('{play_path}')")
        self.con.execute(f"CREATE TABLE players AS SELECT * FROM read_csv_auto('{players_path}')")
        
        # 攻撃数の取得
        self.attack_count = self.con.execute("SELECT COUNT(DISTINCT (攻撃履歴No, 試合ID)) FROM play WHERE 攻撃履歴No > 0").fetchone()[0]
        
        # 選手ID辞書の作成
        self._create_player_id_mapping()
    
    def _create_player_id_mapping(self):
        """選手ID辞書を作成: (試合ID, HA, 背番号) → 選手ID"""
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
            
            # 辞書の作成
            self.game_ha_no_to_playerid = {}
            for _, row in players_df.iterrows():
                try:
                    key = (int(row['GameID']), int(row['HA']), int(row['No']))
                    self.game_ha_no_to_playerid[key] = row['選手ID']
                except Exception as e:
                    print(f"[WARN] players.csv rowアクセス失敗: {e}, row={row}")
                    
        except Exception as e:
            print(f"[WARN] playersテーブルの読み込みに失敗: {e}")
            self.game_ha_no_to_playerid = {}

    def get_attack_batch(self, attack_no=None, game_id=None):
        """
        攻撃データを取得
        
        Args:
            attack_no: 攻撃履歴No（Noneの場合はランダム選択）
            game_id: 試合ID（Noneの場合はランダム選択）
            
        Returns:
            dict: 攻撃データ（Noneの場合はデータなし）
        """
        # 攻撃履歴Noと試合IDの取得
        attacks = self.con.execute("SELECT DISTINCT 攻撃履歴No, 試合ID FROM play WHERE 攻撃履歴No > 0").df()
        if attack_no is None or game_id is None:
            attack = attacks.sample(1).iloc[0]
            attack_no = attack['攻撃履歴No']
            game_id = attack['試合ID']
        
        # フレーム範囲の特定
        play_rows = self.con.execute(f"SELECT フレーム番号 FROM play WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} ORDER BY フレーム番号").df()
        if play_rows.empty:
            print(f"[DEBUG] フレームデータなし: 試合{game_id} 攻撃{attack_no}")
            return None
        
        min_frame = play_rows['フレーム番号'].min()
        max_frame = play_rows['フレーム番号'].max()
        
        # トラッキングデータの抽出
        tracking_df = self.con.execute(f"SELECT * FROM tracking WHERE GameID={game_id} AND Frame >= {min_frame} AND Frame <= {max_frame} ORDER BY Frame").df()
        
        if tracking_df.empty:
            print(f"[DEBUG] トラッキングデータなし: 試合{game_id} 攻撃{attack_no} (フレーム範囲: {min_frame}-{max_frame})")
            return None
        
        # 選手別グループ化
        ha_no_groups = {}
        for (ha, no), group in tracking_df.groupby(['HA', 'No']):
            ha_no_groups[(ha, no)] = group.reset_index(drop=True)
        
        if not ha_no_groups:
            print(f"[DEBUG] 選手データなし: 試合{game_id} 攻撃{attack_no}")
            return None
        
        return {
            'attack_no': attack_no,
            'game_id': game_id,
            'min_frame': min_frame,
            'max_frame': max_frame,
            'tracking_frames': tracking_df,
            'ha_no_groups': ha_no_groups,
            'play_rows': self.con.execute(f"SELECT * FROM play WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} ORDER BY フレーム番号").df()
        }

    def get_attack_tensor(self, attack_no=None, game_id=None):
        """
        攻撃データをテンソル形式に変換
        
        Args:
            attack_no: 攻撃履歴No
            game_id: 試合ID
            
        Returns:
            tuple: (tensor, entities, frames)
                - tensor: (23, フレーム数, 2) のnumpy配列
                - entities: 選手IDリスト
                - frames: フレーム番号リスト
        """
        batch = self.get_attack_batch(attack_no, game_id)
        
        if batch is None:
            raise ValueError(f"攻撃データの取得に失敗: attack_no={attack_no}, game_id={game_id}")
        
        ha_no_groups = batch['ha_no_groups']
        frames = batch['tracking_frames']['Frame'].unique()
        frames.sort()
        
        # 選手交代情報の取得
        substitution_info = self._get_substitution_info(batch['game_id'], batch['attack_no'])
        
        # 選手の選択とフィルタリング
        entity_keys = self._select_entities(ha_no_groups, substitution_info, batch['game_id'])
        
        # 選手IDの取得
        entities = self._get_entity_ids(entity_keys, ha_no_groups)
        
        # テンソルの作成
        tensor = self._create_tensor(entity_keys, ha_no_groups, frames)
        
        return tensor, entities, frames

    def _get_substitution_info(self, game_id, attack_no):
        """選手交代情報を取得"""
        try:
            # フレーム範囲の取得
            play_rows = self.con.execute(f"""
                SELECT フレーム番号 FROM play 
                WHERE 攻撃履歴No={attack_no} AND 試合ID={game_id} 
                ORDER BY フレーム番号
            """).df()
            
            if play_rows.empty:
                return {}
            
            min_frame = play_rows['フレーム番号'].min()
            max_frame = play_rows['フレーム番号'].max()
            
            # 交代情報の取得
            substitutions = self.con.execute(f"""
                SELECT フレーム番号, 選手名, 選手背番号, ホームアウェイF, チームID
                FROM play 
                WHERE 試合ID={game_id} 
                AND フレーム番号 >= {min_frame} 
                AND フレーム番号 <= {max_frame}
                AND アクション名 LIKE '%交代%'
                ORDER BY フレーム番号
            """).df()
            
            # 交代情報の辞書化
            substitution_info = {}
            for _, sub in substitutions.iterrows():
                key = (sub['ホームアウェイF'], sub['選手背番号'])
                if key not in substitution_info:
                    substitution_info[key] = {
                        'player_name': sub['選手名'],
                        'team_id': sub['チームID'],
                        'substitution_frame': sub['フレーム番号'],
                        'is_substituted_out': True
                    }
            
            return substitution_info
            
        except Exception as e:
            print(f"[WARN] 交代情報の取得に失敗: {e}")
            return {}

    def _select_entities(self, ha_no_groups, substitution_info, game_id):
        """エンティティ（選手・ボール）を選択"""
        # 実際に存在する選手番号を取得
        existing_players = list(ha_no_groups.keys())
        
        # ボールと選手を分離
        ball_key = (0, 0) if (0, 0) in ha_no_groups else None
        player_keys = [key for key in existing_players if key != (0, 0)]
        
        # 交代選手の除外
        filtered_player_keys = self._filter_players_by_substitution(
            player_keys, substitution_info, game_id
        )
        
        # 最大22選手まで選択
        max_players = 22
        if len(filtered_player_keys) > max_players:
            print(f"[WARN] 選手数が多すぎます: {len(filtered_player_keys)} > {max_players}")
            filtered_player_keys = self._select_most_frequent_players(
                filtered_player_keys, ha_no_groups, max_players
            )
        
        # entity_keysの生成
        entity_keys = []
        if ball_key:
            entity_keys.append(ball_key)
        entity_keys.extend(filtered_player_keys)
        
        # 23エンティティに満たない場合はNoneで埋める
        while len(entity_keys) < 23:
            entity_keys.append(None)
        
        return entity_keys

    def _filter_players_by_substitution(self, player_keys, substitution_info, game_id):
        """交代情報に基づいて選手を除外"""
        if not substitution_info:
            return player_keys
        
        filtered_keys = []
        for key in player_keys:
            if key not in substitution_info:
                filtered_keys.append(key)
            else:
                print(f"[INFO] 交代選手を除外: HA={key[0]}, No={key[1]}, 選手名={substitution_info[key]['player_name']}")
        
        return filtered_keys

    def _select_most_frequent_players(self, player_keys, ha_no_groups, max_players):
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

    def _get_entity_ids(self, entity_keys, ha_no_groups):
        """エンティティの選手IDを取得"""
        entity_id_map = {}
        
        for i, (ha, no) in enumerate(entity_keys):
            if (ha, no) is None:
                entity_id_map[i] = None
            elif (ha, no) == (0, 0):
                entity_id_map[i] = 'ball'
            elif (ha, no) in ha_no_groups and not ha_no_groups[(ha, no)].empty:
                df = ha_no_groups[(ha, no)]
                game_id0 = df.iloc[0]['GameID']
                
                # 選手IDの取得
                key = (int(game_id0), int(ha), int(no))
                pid = self.game_ha_no_to_playerid.get(key, None)
                if pid is None:
                    alt_key = (game_id0, ha, no)
                    pid = self.game_ha_no_to_playerid.get(alt_key, None)
                
                entity_id_map[i] = pid
            else:
                entity_id_map[i] = None
        
        return [entity_id_map[i] for i in range(23)]

    def _create_tensor(self, entity_keys, ha_no_groups, frames):
        """テンソルを作成"""
        tensor = np.full((23, len(frames), 2), np.nan, dtype=np.float32)
        frame_to_idx = {f: i for i, f in enumerate(frames)}
        
        for ent_idx, (ha, no) in enumerate(entity_keys):
            if (ha, no) is not None and (ha, no) in ha_no_groups:
                df = ha_no_groups[(ha, no)]
                for _, row in df.iterrows():
                    fidx = frame_to_idx[row['Frame']]
                    tensor[ent_idx, fidx, 0] = row['X']
                    tensor[ent_idx, fidx, 1] = row['Y']
        
        return tensor

    def save_all_attack_data(self, save_dir="Preprocessed_data", format="parquet"):
        """
        全攻撃データを保存
        
        Args:
            save_dir: 保存ディレクトリ
            format: 保存形式 ("parquet" または "feather")
        """
        import os
        
        # 保存ディレクトリの作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 全攻撃データの取得
        all_attacks = self.con.execute("""
            SELECT DISTINCT 試合ID, 攻撃履歴No 
            FROM play 
            WHERE 攻撃履歴No > 0
            ORDER BY 試合ID, 攻撃履歴No
        """).df()
        
        print(f"保存開始: {len(all_attacks)}個の攻撃データ (形式: {format})")
        
        success_count = 0
        error_count = 0
        
        for idx, (_, attack) in enumerate(all_attacks.iterrows()):
            try:
                # 攻撃データの取得
                attack_data = self.get_attack_batch(attack['攻撃履歴No'], attack['試合ID'])
                if attack_data is None:
                    print(f"[WARN] 攻撃データが取得できません: {attack['試合ID']}_{attack['攻撃履歴No']} (フレームデータなし)")
                    error_count += 1
                    continue
                
                # テンソルデータの取得
                try:
                    tensor, entities, frames = self.get_attack_tensor(
                        attack['攻撃履歴No'], 
                        attack['試合ID']
                    )
                except (ValueError, Exception) as e:
                    print(f"[WARN] テンソル生成に失敗: {attack['試合ID']}_{attack['攻撃履歴No']} - {e}")
                    error_count += 1
                    continue
                
                # データの保存
                self._save_attack_files(attack, attack_data, tensor, entities, frames, save_dir, format)
                
                success_count += 1
                
                # 進捗表示
                if (idx + 1) % 100 == 0:
                    print(f"保存進捗: {idx + 1}/{len(all_attacks)} (成功: {success_count}, エラー: {error_count})")
                    
            except Exception as e:
                print(f"エラー: 攻撃{attack['試合ID']}_{attack['攻撃履歴No']}の保存に失敗: {e}")
                error_count += 1
                continue
        
        print(f"保存完了: {save_dir}")
        print(f"成功: {success_count}, エラー: {error_count}, 総数: {len(all_attacks)}")

    def _save_attack_files(self, attack, attack_data, tensor, entities, frames, save_dir, format):
        """攻撃データをファイルとして保存"""
        import os
        
        # ファイル名の生成
        filename = f"attack_{attack['試合ID']}_{attack['攻撃履歴No']:04d}"
        
        # テンソルデータのDataFrame変換
        tensor_df = pd.DataFrame({
            'entity_idx': np.repeat(range(tensor.shape[0]), tensor.shape[1]),
            'frame_idx': np.tile(range(tensor.shape[1]), tensor.shape[0]),
            'x': tensor.reshape(-1, 2)[:, 0],
            'y': tensor.reshape(-1, 2)[:, 1]
        })
        
        # メタデータのDataFrame変換
        meta_df = pd.DataFrame({
            'attack_no': [attack['攻撃履歴No']],
            'game_id': [attack['試合ID']],
            'min_frame': [attack_data['min_frame']],
            'max_frame': [attack_data['max_frame']],
            'frame_count': [len(frames)],
            'tensor_shape_0': [tensor.shape[0]],
            'tensor_shape_1': [tensor.shape[1]],
            'tensor_shape_2': [tensor.shape[2]],
            'entities_count': [len(entities)],
            'entities': [str(entities)],
            'frames': [str(frames.tolist())]
        })
        
        # ディレクトリの作成
        tensor_dir = os.path.join(save_dir, "tensor")
        meta_dir = os.path.join(save_dir, "meta")
        os.makedirs(tensor_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        
        # ファイルの保存
        if format.lower() == "parquet":
            tensor_path = os.path.join(tensor_dir, f"{filename}.parquet")
            meta_path = os.path.join(meta_dir, f"{filename}.parquet")
            
            tensor_df.to_parquet(tensor_path, index=False)
            meta_df.to_parquet(meta_path, index=False)
            
        elif format.lower() == "feather":
            tensor_path = os.path.join(tensor_dir, f"{filename}.feather")
            meta_path = os.path.join(meta_dir, f"{filename}.feather")
            
            tensor_df.to_feather(tensor_path)
            meta_df.to_feather(meta_path)

    def load_attack_data(self, filename, data_dir="Preprocessed_data", format="parquet"):
        """
        保存された攻撃データを読み込み
        
        Args:
            filename: ファイル名（拡張子なし）
            data_dir: データディレクトリ
            format: ファイル形式
            
        Returns:
            dict: 攻撃データ
        """
        import os
        import ast
        
        # ディレクトリパスの構築
        tensor_dir = os.path.join(data_dir, "tensor")
        meta_dir = os.path.join(data_dir, "meta")
        
        # ファイルパスの構築
        if format.lower() == "parquet":
            tensor_path = os.path.join(tensor_dir, f"{filename}.parquet")
            meta_path = os.path.join(meta_dir, f"{filename}.parquet")
        elif format.lower() == "feather":
            tensor_path = os.path.join(tensor_dir, f"{filename}.feather")
            meta_path = os.path.join(meta_dir, f"{filename}.feather")
        
        # ファイルの存在確認
        if not os.path.exists(tensor_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {tensor_path} または {meta_path}")
        
        # データの読み込み
        if format.lower() == "parquet":
            tensor_df = pd.read_parquet(tensor_path)
            meta_df = pd.read_parquet(meta_path)
        elif format.lower() == "feather":
            tensor_df = pd.read_feather(tensor_path)
            meta_df = pd.read_feather(meta_path)
        
        # テンソルの再構築
        shape = (meta_df['tensor_shape_0'].iloc[0], 
                meta_df['tensor_shape_1'].iloc[0], 
                meta_df['tensor_shape_2'].iloc[0])
        
        tensor = np.full(shape, np.nan, dtype=np.float32)
        for _, row in tensor_df.iterrows():
            entity_idx = int(row['entity_idx'])
            frame_idx = int(row['frame_idx'])
            tensor[entity_idx, frame_idx, 0] = row['x']
            tensor[entity_idx, frame_idx, 1] = row['y']
        
        # entitiesとframesの復元
        entities = ast.literal_eval(meta_df['entities'].iloc[0])
        frames = np.array(ast.literal_eval(meta_df['frames'].iloc[0]))
        
        # データの返却
        return {
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

    def close(self):
        """データベース接続を閉じる"""
        self.con.close()

if __name__ == "__main__":
    # データローダーの初期化
    loader = BatchDataLoader(
        tracking_path="../../data/2022_data/*/tracking.csv",
        play_path="../../data/2022_data/*/play.csv",
        players_path="../../data/2022_data/*/players.csv"
    )

    print(f"総攻撃数: {loader.attack_count}")
    
    # Parquet形式で保存
    print(f"\n=== Parquet形式で保存 ===")
    loader.save_all_attack_data("Preprocessed_data/parquet", format="parquet")
    
    # Feather形式で保存
    print(f"\n=== Feather形式で保存 ===")
    loader.save_all_attack_data("Preprocessed_data/feather", format="feather")
    
    # 読み込みテスト
    print(f"\n=== 読み込みテスト ===")
    try:
        data = loader.load_attack_data("attack_2022091601_0001", "Preprocessed_data/parquet", format="parquet")
        print(f"Parquet読み込み成功")
        print(f"  テンソルshape: {data['tensor'].shape}")
        print(f"  選手ID: {data['entities']}")
        print(f"  フレーム数: {len(data['frames'])}")
        print(f"  情報: {data['info']}")
    except Exception as e:
        print(f"Parquet読み込みエラー: {e}")
    
    try:
        data = loader.load_attack_data("attack_2022091601_0001", "Preprocessed_data/feather", format="feather")
        print(f"Feather読み込み成功")
        print(f"  テンソルshape: {data['tensor'].shape}")
        print(f"  選手ID: {data['entities']}")
        print(f"  フレーム数: {len(data['frames'])}")
        print(f"  情報: {data['info']}")
    except Exception as e:
        print(f"Feather読み込みエラー: {e}")
    
    loader.close()