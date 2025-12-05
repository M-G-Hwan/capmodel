import pandas as pd
import os

def balance_dataset():
    # 1. 파일 경로 (사용자님이 주신 경로 그대로 유지)
    csv_path = 'datacollector/dataset_modified/data_labels.csv' 
    save_path = 'datacollector/dataset_modified/balanced_data_labels.csv'

    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        # 헤더가 있다고 가정하고 읽어봅니다.
        df = pd.read_csv(csv_path)
        print(f"데이터 로드 완료. 발견된 컬럼 이름들: {list(df.columns)}")
    except Exception as e:
        print(f"CSV 읽기 실패: {e}")
        return

    # 2. '각도' 컬럼 자동 탐지 (Smart Detect)
    # 150도는 여기서 이미 빠져있습니다 (0, 40, 90, 120만 타겟)
    target_angles = [0, 40, 90, 120]
    angle_col_name = None

    print("\n[각도 컬럼 자동 탐색 중...]")
    for col in df.columns:
        unique_vals = df[col].unique()
        try:
            numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna().unique()
            if any(val in target_angles for val in numeric_vals):
                 if 90 in numeric_vals or 120 in numeric_vals or 0 in numeric_vals:
                    angle_col_name = col
                    print(f"-> 찾았다! '{col}' 컬럼이 각도 데이터입니다.")
                    break
        except:
            continue
    
    if angle_col_name is None:
        print("경로: 자동 탐색 실패. 'angle'이라는 이름이 포함된 컬럼을 찾습니다.")
        for col in df.columns:
            if 'angle' in str(col).lower():
                angle_col_name = col
                break
    
    if angle_col_name is None:
        print("❌ 실패: 각도 데이터를 찾을 수 없습니다.")
        return

    # 3. 데이터 분포 확인
    print(f"\n--- ['{angle_col_name}'] 컬럼 분포 확인 ---")
    print(df[angle_col_name].value_counts())

    # 4. 밸런싱 시작 (★ 여기가 수정된 핵심 파트입니다!)
    # 기존: min_count (최소값) 찾기 -> 수정: 40도 개수 찾기
    
    # 40도 데이터 개수 확인 (우리의 기준점)
    group_40 = df[df[angle_col_name] == 40]
    target_count = len(group_40)
    
    if target_count == 0:
        print("오류: 기준이 될 40도 데이터가 하나도 없습니다!")
        return

    print(f"\n>> 기준: 40도 데이터 ({target_count}개)에 맞춰서 나머지를 뻥튀기합니다!\n")

    balanced_dfs = []
    
    # 설정한 각도(0, 40, 90, 120)만 돌면서 작업
    for angle in target_angles:
        group = df[df[angle_col_name] == angle]
        count = len(group)
        
        if count == 0:
            print(f"  -> {angle}도 데이터 없음 (패스)")
            continue
            
        # ★ 핵심 로직: 개수가 40도보다 적으면 복사(replace=True)한다!
        # 같거나 많으면 그냥 뽑거나 랜덤으로 줄인다(replace=False)
        is_replace = (count < target_count)
        
        sampled_group = group.sample(n=target_count, replace=is_replace, random_state=42)
        balanced_dfs.append(sampled_group)
        
        status = "복사해서 늘림 🔼" if is_replace else "개수 맞춤/유지 ⏺️"
        print(f"  -> {angle}도: {count}개 -> {target_count}개 ({status})")

    # 5. 저장하기
    if balanced_dfs:
        final_df = pd.concat(balanced_dfs)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 헤더 없이 저장
        final_df.to_csv(save_path, index=False, header=False)
        
        print("\n--- [최종 결과 분포] ---")
        print(final_df[angle_col_name].value_counts())
        print(f"\n✅ 성공! 저장 완료: {save_path}")
    else:
        print("결과를 생성하지 못했습니다.")

if __name__ == "__main__":
    balance_dataset()