# -*- coding: utf-8 -*-
"""소프트웨어_시간표.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19CHjysKpZxgQa1s2qs09NetR9VGaHl8X
"""

# 두 조건을 모두 만족하였지만 한번씩 같은 수업이 일주일에 3개씩 있는 오류가 발생함.

import pandas as pd
import random

# 과목 및 교수님 리스트를 만듭니다.
subjects = ['확률과통계/고선우', '딥러닝/고선우', 'AI알고리즘/권수태', '기계학습/권수태',
            'AI수학/민정익', '논리적문제해결1/민정익', '인공지능기초/이근호', '서비스러닝/이근호',
            '논리적문제해결2/송주환', '영상이해/송주환', 'AI프로그래밍/김영수', '리눅스운영체제/김영수']

# 시간대 리스트
time_slots = ['09:00', '10:30', '12:00', '01:00', '02:30', '04:00']

# 무작위로 시간표를 생성
timetable = pd.DataFrame(index=time_slots, columns=['월', '화', '수', '목', '금'])

# 각 요일에 배정된 과목을 기록하는 변수
assigned_subjects = {'월': set(), '화': set(), '수': set(), '목': set(), '금': set()}

# 이전 요일에 배정된 과목을 추적하기 위한 변수
prev_day_subjects = {}

def is_valid_assignment(day, time, subject):
    # 이전 요일에 같은 과목이 배정되지 않도록 확인
    prev_day = '월' if day == '화' else '화' if day == '수' else '수' if day == '목' else '목' if day == '금' else '금'

    # prev_day에 해당 과목이 이미 배당되지 않았을 경우에만 True를 반환
    return subject not in prev_day_subjects.get(prev_day, set())

for day in timetable.columns:
    for time in timetable.index:
        # if조건은 현재 시간이 12시인 경우 점심시간을 할당
        if time == '12:00':
            timetable.loc[time, day] = "점심시간"

  # elif조건은 요일이 수 이고 시간이 4시인 경우 진로탐색을 할당
        elif day == '수' and time == '04:00':
            timetable.loc[time, day] = "진로탐색"

  # 해당 요일에 이미 할당된 과목을 제외한 모든 과목을 available_subjects에 저장 = 이미 배정된 과목이 중복 x
        else:
            available_subjects = set(subjects) - assigned_subjects[day]

  # 해당 요일에 아직 할당되지 않은 과목들이 있는지 확인
            if available_subjects:

              # valid_subjects라는 리스트를 생성 후 현재 요일과 시간, 그리고 각 과목에 대한 충돌여부 검사
                valid_subjects = [subject for subject in available_subjects if is_valid_assignment(day, time, subject)]

              # 이 조건은 유효한 과목들이 존재하는 경우에만 실행됨.
                if valid_subjects:
                  # 유효한 과목 중에서 무작위로 하나의 과목을 선택하여 subject 변수에 할당
                    subject = random.choice(valid_subjects)

              # 이 부분은 이전 코드에서 유효한 과목을 찾지 못한 경우 실행
                else:
                  # 남아 있는 모든 과목중 하나를 무작위로 선택 후 subject에 할당 but 이 과목은 유효한지 판단여부 X
                    subject = random.choice(list(available_subjects))

                # 선택된 과목을 해당 요일과 시간대에 할당
                # 이 코드에선 유효성 검사를 거치지 않고 남은 모든 과목 중에서 무작위 선택하여 할당
                timetable.loc[time, day] = subject

                # assigned_subjects 집합에 선택된 과목 추가
                # 이렇게 하면 같은 요일에 동일한 과목이 중복되서 할당되지 않음.
                assigned_subjects[day].add(subject)

        # 현재 요일의 과목을 이전 요일의 과목으로 설정
        prev_day_subjects[day] = assigned_subjects[day]

# 생성된 시간표를 출력
timetable