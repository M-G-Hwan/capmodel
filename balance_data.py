import pandas as pd
import os

def balance_dataset_to_optimal():
    # 1. 경로 설정
    base_dir = 'datacollector/dataset'
    csv_file = 'data_labels.csv'
    save_file = 'balanced_data_labels.csv'
    
    # [핵심] 모든 각도를 6,000개로 통일합니다.
    target_count = 6000 

    csv_path = os.path.join(base_dir, csv_file)
    save_path = os.path.join(base_dir, save_file)

    # 2. 데이터 불러오기 및 정확한 컬럼 매핑
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        # 파일의 4개 컬럼을 정확히 읽습니다. (헤더가 있다면 헤더는 무시하고 첫 줄부터 데이터로 처리)
        df = pd.read_csv(csv_path, 
                         header=None, # 헤더 행이 존재하더라도 데이터로 처리하거나, 헤더 행이 없다면 정상적으로 처리
                         names=['timestamp', 'filename', 'angle', 'speed'], 
                         dtype=str)
        
        # 'angle' (servo_angle) 컬럼을 숫자로 변환
        df['angle'] = pd.to_numeric(df['angle'], errors='coerce')
        df = df.dropna(subset=['angle']) 
        df['angle'] = df['angle'].astype(int)
        
        print(f"1. 데이터 로드 완료: {len(df)}개")
    except Exception as e:
        print(f"CSV 읽기 실패: {e}")
        return

    # 3. 파일 유효성 검사 (실제 파일 없는 행 삭제)
    print("\n[클리닝] 파일 존재 여부 확인 중...")
    def check_file_exists(row):
        # filename은 이미 'image_path'를 포함하고 있습니다.
        file_name = str(row['filename']).strip()
        # image_path가 'path/to/image.png' 형태일 때, base_dir이 'datacollector/dataset'라면 경로 조정 필요
        # 일반적으로 data_labels.csv 내의 path는 base_dir을 기준으로 상대 경로입니다.
        full_path = os.path.join(base_dir, file_name) 
        return os.path.exists(full_path)

    valid_mask = df.apply(check_file_exists, axis=1)
    df_clean = df[valid_mask].copy()
    print(f" -> 유효 데이터: {len(df_clean)}개 (삭제됨: {len(df)-len(df_clean)}개)")

    # 4. 밸런싱 (모두 6,000개로 맞춤)
    print(f"\n[밸런싱] 모든 각도를 {target_count}개로 통일합니다...")
    
    balanced_dfs = []
    angles = df_clean['angle'].unique()
    
    for angle in angles:
        group = df_clean[df_clean['angle'] == angle]
        count = len(group)
        
        if count > 0:
            replace_flag = count < target_count
            group = group.sample(n=target_count, replace=replace_flag, random_state=42)
            
        balanced_dfs.append(group)

    final_df = pd.concat(balanced_dfs).sort_index()

    # 5. 결과 확인
    print("\n-----------------------------")
    print("   [최종 데이터 분포]   ")
    print("-----------------------------")
    if not final_df.empty:
        print(final_df['angle'].value_counts().sort_index())
    print("-----------------------------")
    print(f"총 데이터 개수: {len(final_df)}")

    # 6. 저장 (timestamp 컬럼은 제외하고 저장)
    if not final_df.empty:
        # 'timestamp' 컬럼을 제외하고 'filename', 'angle', 'speed'만 저장합니다.
        final_df[['filename', 'angle', 'speed']].to_csv(save_path, index=False, header=False)
        print(f"\n✅ 저장 완료: {save_file}")

if __name__ == "__main__":
    balance_dataset_to_optimal()