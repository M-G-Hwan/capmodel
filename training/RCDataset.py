# training/RCDataset.py

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor


class RCDataset(Dataset):
    """
    RC 자율주행용 Dataset

    - CSV 안에 'split' 컬럼이 있으면 그대로 사용
    - 없으면 servo_angle별 stratified train/test split 수행
    - __getitem__ 에서:
        - 이미지 BGR 로드
        - (train일 때만) 증강
        - RCPreprocessor로 전처리
        - 라벨을 클래스 인덱스로 변환
    """

    def __init__(self,
                 csv_filename,
                 root,
                 preprocessor: RCPreprocessor,
                 augmentor: RCAugmentor = None,
                 split: str = "train",
                 split_ratio: float = 0.8,
                 shuffle: bool = True,
                 random_seed: int = 42):

        self.image_root = root
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split

        csv_path = os.path.join(root, csv_filename)
        self.df_full = pd.read_csv(csv_path)


        # ============================================================
        # [추가할 코드] 실제 파일이 없는 데이터는 리스트에서 삭제하기
        # ============================================================
        print(f"[Check] 데이터 검증 시작... (총 {len(self.df_full)}개)")
        
        # 파일이 진짜 있는지 확인하는 함수
        def check_file_exists(row):
            # 경로에서 잡다한 거 떼고 '파일명'만 딱 가져오기
            filename = os.path.basename(str(row["image_path"]))
            # 전체 경로 만들기 (root + 파일명)
            full_path = os.path.join(self.image_root, filename)
            return os.path.exists(full_path)

        # 없는 파일 걸러내기 (mask가 True인 것만 남김)
        mask = self.df_full.apply(check_file_exists, axis=1)
        removed_count = len(self.df_full) - mask.sum()
        
        if removed_count > 0:
            self.df_full = self.df_full[mask].reset_index(drop=True)
            print(f"[Warning] 🚨 실제 사진이 없는 {removed_count}개 데이터를 목록에서 삭제했습니다.")
            print(f"-> 남은 데이터: {len(self.df_full)}개")
        else:
            print("[Info] ✅ 모든 데이터 파일이 정상적으로 존재합니다.")
        # ============================================================


        # ------------------------------------------------------
        # 1) CSV에 split 컬럼이 있으면 그대로 사용
        # ------------------------------------------------------
        if "split" in self.df_full.columns:
            print("[RCDataset] Using existing 'split' column from CSV.")
            self.df = self.df_full[self.df_full["split"] == split].reset_index(drop=True)

        # ------------------------------------------------------
        # 2) 없으면 servo_angle별 stratified split
        # ------------------------------------------------------
        else:
            print("[RCDataset] Performing stratified split...")

            if shuffle:
                self.df_full = self.df_full.sample(frac=1.0, random_state=random_seed)

            df_list = []
            for angle, df_group in self.df_full.groupby("servo_angle"):
                n = len(df_group)
                n_train = int(n * split_ratio)

                if split == "train":
                    df_split = df_group.iloc[:n_train]
                else:
                    df_split = df_group.iloc[n_train:]

                df_list.append(df_split)

            self.df = pd.concat(df_list).reset_index(drop=True)

        # ------------------------------------------------------
        # 3) angle → class index 매핑
        # ------------------------------------------------------
        self.angles = sorted(self.df["servo_angle"].unique().tolist())
        self.angle_to_idx = {a: i for i, a in enumerate(self.angles)}

        print(f"[RCDataset:{split}] samples={len(self.df)}, per-class counts:")
        print(self.df["servo_angle"].value_counts().sort_index())

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------------------------------------
    # 이 부분(def __getitem__)을 이걸로 통째로 바꿔줘!
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) CSV에서 파일 경로 가져오기
        raw_path = str(row["image_path"])
        
        # 2) 안전장치: CSV 경로에 폴더명이 섞여있어도 "파일명(abc.png)"만 추출
        filename = os.path.basename(raw_path)
        
        # 3) 진짜 경로 만들기 (root 경로 + 파일명)
        img_path = os.path.join(self.image_root, filename)

        # 4) 이미지 로드 (BGR)
        img_bgr = cv2.imread(img_path)
        
        # 로드 실패 시 디버깅 메시지 출력 (이게 중요!)
        if img_bgr is None:
            print(f"\n[!!! Error !!!]")
            print(f"이미지를 못 읽었습니다.")
            print(f"1. CSV에 적힌 내용: {raw_path}")
            print(f"2. 프로그램이 찾은 경로: {img_path}")
            print(f"-> 경로가 실제 파일 위치와 맞는지 확인해주세요.\n")
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 5) train split일 때만 증강
        angle = int(row["servo_angle"])
        if self.split == "train" and self.augmentor is not None:
            img_bgr, angle = self.augmentor(img_bgr, angle)

        # 6) 공통 전처리 (추론과 동일)
        img_chw = self.preprocessor(img_bgr)       # np.ndarray, (3, 66, 200), float32
        img_tensor = torch.from_numpy(img_chw).float()

        # 7) 라벨을 클래스 인덱스로 변환
        label = self.angle_to_idx[angle]

        return img_tensor, label


if __name__ == "__main__":
    preproc = RCPreprocessor(out_size=(200, 66),
                             crop_top_ratio=0.4,
                             crop_bottom_ratio=1.0)
    augment = RCAugmentor()

    dataset = RCDataset(
        csv_filename="balanced_data_labels.csv",
        root="datacollector/dataset_modified",
        preprocessor=preproc,
        augmentor=augment,
        split="train",
        split_ratio=0.8,
    )

    x, y = dataset[0]
    print("img shape:", x.shape, "label:", y)
