import pandas as pd
import os

def clean_and_balance_dataset_final_v2():
    # 1. 설정
    base_dir = 'datacollector/dataset' # 폴더 경로 확인
    csv_file = 'data_labels.csv'
    save_file = 'balanced_data_labels.csv'
    
    max_samples = 2000   # 일반 데이터 제한
    target_150 = 500     # 150도 데이터 목표 (증강)

    csv_path = os.path.join(base_dir, csv_file)
    save_path = os.path.join(base_dir, save_file)

    # 2. 데이터 불러오기
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        # 일단 모든 것을 문자로 읽어들입니다.
        df = pd.read_csv(csv_path, header=None, names=['filename', 'angle', 'speed'], dtype=str)
        
        # [핵심 수정] 'angle' 컬럼을 숫자로 변환 시도
        # errors='coerce' -> 숫자가 아닌 글자('servo_angle' 등)는 NaN(빈값)으로 바꿔버림
        df['angle'] = pd.to_numeric(df['angle'], errors='coerce')
        
        # NaN(빈값)이 된 행(=원래 제목이었던 행)을 삭제
        before_len = len(df)
        df = df.dropna(subset=['angle'])
        after_len = len(df)
        
        if before_len > after_len:
            print(f"⚠️ 제목(Header) 혹은 잘못된 데이터 {before_len - after_len}개 행을 삭제했습니다.")

        # 이제 안전하게 정수로 변환
        df['angle'] = df['angle'].astype(int)
        
        print(f"1. 데이터 로드 및 정제 완료: {len(df)}개")
        
    except Exception as e:
        print(f"CSV 처리 중 오류 발생: {e}")
        return

    # 3. 이미지 파일 존재 여부 확인 (클리닝)
    print("\n[클리닝] 파일 존재 여부 검사 중...")
    def check_file_exists(row):
        file_name = str(row['filename']).strip()
        if not file_name.endswith('.png'):
            file_name += '.png'
        full_path = os.path.join(base_dir, file_name)
        return os.path.exists(full_path)

    valid_mask = df.apply(check_file_exists, axis=1)
    df_clean = df[valid_mask].copy()
    print(f" -> 유효 데이터: {len(df_clean)}개")

    # 4. 밸런싱 (2000개 자르기 & 150도 500개로 늘리기)
    print(f"\n[밸런싱] 150도는 {target_150}개로 증강, 나머지는 {max_samples}개 제한...")
    
    balanced_dfs = []
    # 각도가 존재하는지 확인
    if 'angle' in df_clean.columns:
        angles = df_clean['angle'].unique()
        
        for angle in angles:
            group = df_clean[df_clean['angle'] == angle]
            count = len(group)
            
            if angle == 150:
                print(f"✨ 150도 발견! ({count}개) -> {target_150}개로 증강합니다.")
                group = group.sample(n=target_150, replace=True, random_state=42)
                
            elif count > max_samples:
                group = group.sample(n=max_samples, replace=False, random_state=42)
                
            balanced_dfs.append(group)
        
        if balanced_dfs:
            final_df = pd.concat(balanced_dfs).sort_index()
        else:
            final_df = pd.DataFrame()
    else:
        print("오류: 데이터프레임에 angle 컬럼이 없습니다.")
        return

    # 5. 결과 확인
    print("\n-----------------------------")
    print("   [최종 각도별 분포]   ")
    print("-----------------------------")
    if not final_df.empty:
        print(final_df['angle'].value_counts().sort_index())
    print("-----------------------------")
    
    # 6. 저장
    if not final_df.empty:
        final_df.to_csv(save_path, index=False, header=False)
        print(f"\n✅ 저장 완료: {save_file} (총 {len(final_df)}개)")

if __name__ == "__main__":
    clean_and_balance_dataset_final_v2()