import os
import re
import pandas as pd

# 1. 이미지 파일이 들어있는 폴더 경로 (사용자 환경에 맞게 수정)
folder_path = 'datacollector/dataset' 

# 2. 데이터를 저장할 리스트
angle_counts = []

# 3. 폴더 내의 모든 파일을 하나씩 확인
try:
    file_list = os.listdir(folder_path)
    
    for filename in file_list:
        # png 파일인 경우만 처리 (jpg면 .jpg로 변경)
        if filename.endswith('.png'):
            
            # 정규표현식(Regex)을 사용해 'angle' 뒤에 붙은 숫자 추출
            # 예: ..._angle90_...  -> '90'을 찾음
            match = re.search(r'angle(\d+)', filename)
            
            if match:
                angle_value = int(match.group(1)) # 숫자 형태로 변환
                angle_counts.append(angle_value)
            else:
                # angle 패턴이 없는 파일이 있다면 출력해보기 (확인용)
                pass 

    # 4. 결과 정리 및 출력
    if angle_counts:
        # 판다스를 이용해 보기 좋게 개수 세기
        df = pd.DataFrame(angle_counts, columns=['Angle'])
        
        # 각도별 개수 세기 (sort_index로 각도 순서대로 정렬)
        result = df['Angle'].value_counts().sort_index()
        
        print("-" * 30)
        print("   각도(Angle)  |   파일 개수")
        print("-" * 30)
        print(result)
        print("-" * 30)
        print(f"총 이미지 개수: {len(df)}")
        
    else:
        print("해당 폴더에서 'angle'이 포함된 png 파일을 찾지 못했습니다.")

except FileNotFoundError:
    print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")