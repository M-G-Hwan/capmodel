import pandas as pd
import os

def clean_dataset():
    # 1. 경로 설정 (사용자님 환경에 맞춤)
    base_dir = 'datacollector/dataset'  # CSV와 이미지가 있는 폴더
    csv_file = 'data_labels.csv'
    save_file = 'balanced_data_labels.csv' # 파일명은 그대로 유지 (내용은 클린본)

    csv_path = os.path.join(base_dir, csv_file)
    save_path = os.path.join(base_dir, save_file)

    # 2. 데이터 불러오기
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return

    try:
        # 헤더가 없으므로 컬럼명을 직접 지정해서 불러옵니다.
        df = pd.read_csv(csv_path, header=None, names=['filename', 'angle', 'speed'])
        print(f"원본 데이터 로드 완료: {len(df)}개")
    except Exception as e:
        print(f"CSV 읽기 실패: {e}")
        return

    # 3. 클리닝 로직 (파일이 실제로 있는지 검사)
    print("\n[클리닝 시작] 실제 이미지 파일이 없는 데이터를 삭제합니다...")

    def check_file_exists(row):
        file_name = str(row['filename'])
        
        # 만약 CSV에 .png가 빠져있다면 붙여줍니다.
        if not file_name.endswith('.png'):
            file_name += '.png'
            
        # 전체 경로를 만들어서 파일이 있는지 확인
        full_path = os.path.join(base_dir, file_name)
        return os.path.exists(full_path)

    # apply 함수를 이용해 유효한 데이터만 남김
    # (이 과정이 데이터 양에 따라 몇 초 걸릴 수 있습니다)
    valid_mask = df.apply(check_file_exists, axis=1)
    df_clean = df[valid_mask]

    deleted_count = len(df) - len(df_clean)

    # 4. 결과 출력 및 저장
    print(f"검사 완료!")
    print(f" - 유지된 데이터: {len(df_clean)}개")
    print(f" - 삭제된 데이터(파일 없음): {deleted_count}개")

    if not df_clean.empty:
        # 인덱스 재정렬 후 저장 (헤더 없이)
        df_clean.to_csv(save_path, index=False, header=False)
        print(f"\n✅ 저장 완료: {save_path}")
    else:
        print("\n❌ 경고: 남은 데이터가 하나도 없습니다. 경로를 다시 확인해주세요.")

if __name__ == "__main__":
    clean_dataset()