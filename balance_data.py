import pandas as pd
import os

def balance_dataset_to_target():
    # 1. 설정
    base_dir = 'datacollector/dataset' 
    csv_file = 'data_labels.csv'
    save_file = 'balanced_data_labels.csv'
    
    # [핵심] 모든 각도를 이 숫자로 통일합니다.
    target_count = 2466 

    csv_path = os.path.join(base_dir, csv_file)
    save_path = os.path.join(base_dir, save_file)

    # 2. 데이터 불러오기 및 에러 처리
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        # 모든 데이터를 문자로 읽어서 'servo_angle' 같은 헤더 오류 방지
        df = pd.read_csv(csv_path, header=None, names=['filename', 'angle', 'speed'], dtype=str)
        
        # angle 컬럼을 숫자로 변환 (변환 안 되는 문자는 NaN 처리 후 삭제)
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
        file_name = str(row['filename']).strip()
        if not file_name.endswith('.png'):
            file_name += '.png'
        full_path = os.path.join(base_dir, file_name)
        return os.path.exists(full_path)

    valid_mask = df.apply(check_file_exists, axis=1)
    df_clean = df[valid_mask].copy()
    print(f" -> 유효 데이터: {len(df_clean)}개 (삭제됨: {len(df)-len(df_clean)}개)")

    # 4. 밸런싱 (모두 2466개로 맞춤)
    print(f"\n[밸런싱] 모든 각도를 {target_count}개로 통일합니다...")
    
    balanced_dfs = []
    angles = df_clean['angle'].unique()
    
    for angle in angles:
        group = df_clean[df_clean['angle'] == angle]
        count = len(group)
        
        if count > target_count:
            # 많으면 -> 자르기 (비복원 추출)
            # print(f" - Angle {angle}: {count} -> {target_count} (축소)")
            group = group.sample(n=target_count, replace=False, random_state=42)
            
        elif count < target_count:
            # 적으면 -> 늘리기 (복원 추출)
            # print(f" - Angle {angle}: {count} -> {target_count} (증강)")
            group = group.sample(n=target_count, replace=True, random_state=42)
        
        else:
            # 딱 맞으면 (거의 없겠지만) 그대로 사용
            pass
            
        balanced_dfs.append(group)

    final_df = pd.concat(balanced_dfs).sort_index()

    # 5. 결과 확인
    print("\n-----------------------------")
    print("   [최종 데이터 분포]   ")
    print("-----------------------------")
    print(final_df['angle'].value_counts().sort_index())
    print("-----------------------------")
    print(f"총 데이터 개수: {len(final_df)}")

    # 6. 저장
    if not final_df.empty:
        final_df.to_csv(save_path, index=False, header=False)
        print(f"\n✅ 저장 완료: {save_file}")

if __name__ == "__main__":
    balance_dataset_to_target()