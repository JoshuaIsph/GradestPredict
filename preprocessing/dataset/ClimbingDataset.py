import torch
import pandas as pd
from torch.utils.data import Dataset


class ClimbingCSVDataset(Dataset):
    def __init__(self, csv_file, all_hold_ids):
        """
        Args:
            csv_file (str): Path to your 'climb_dataset.csv'
            all_hold_ids (list): A sorted list of ALL valid hold IDs on the wall.
                                 (Needed to define the size of the Action Space).
        """
        # 1. Load Data
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['S_LH_ID'] != 'S_LH_ID']
        print(self.df.columns.tolist())


        self.all_holds = all_hold_ids


        # 2. Define Input Columns (The State)
        # These match exactly the keys you saved in your save_dataset.py
        self.feature_cols = [
            'S_LH_x', 'S_LH_y',
            'S_RH_x', 'S_RH_y',
            'S_LF_x', 'S_LF_y',
            'S_RF_x', 'S_RF_y'
        ]
        # Creating a Dictionary to map actions to indices (HD:ID 3, LH -> 12)
        self.limbs = ['LH', 'RH', 'LF', 'RF']
        self.hold_to_index = {}


        counter = 0
        for limb in self.limbs:
            for hold_id in self.all_holds:
                self.hold_to_index[(limb, hold_id)] = counter
                counter += 1
        self.num_actions = counter
        #This converts the Actions from the Databset into the above mapped indices
        self.action_indices = []
        skipped_count = 0

        for idx, row in self.df.iterrows():
            target_hold = int(row['A_Target_ID'])
            limb = row['A_Limb']

            if (limb, target_hold) in self.hold_to_index:
                self.action_indices.append(self.hold_to_index[(limb, target_hold)])
            else:
                # This happens if your CSV has a hold ID that isn't in 'all_hold_ids'
                self.action_indices.append(0)  # Fallback (should be rare)
                skipped_count += 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Extract State Features
        state_features = self.df.iloc[idx][self.feature_cols].values.astype(float)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # 2. Extract Action as Index
        action_index = self.action_indices[idx]
        action_tensor = torch.tensor(action_index, dtype=torch.long)

        return state_tensor, action_tensor





