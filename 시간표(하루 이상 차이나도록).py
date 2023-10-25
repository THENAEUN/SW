#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# 교시와 요일 목록
교시 = ['1교시', '2교시', '3교시', '4교시', '5교시']
요일 = ['월요일', '화요일', '수요일', '목요일', '금요일']

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=object)

# 행과 열에 교시와 요일 할당
for i in range(5):
    for j in range(5):
        matrix[i, j] = f'{교시[i]} - {요일[j]}'

# 행렬 출력
for i in range(5):
    for j in range(5):
        print(matrix[i, j], end='\t')
    print()


# In[2]:


import numpy as np

# 교시와 요일 목록
교시 = ['1교시', '2교시', '3교시', '4교시', '5교시']
요일 = ['월요일', '화요일', '수요일', '목요일', '금요일']

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=object)

# 행과 열에 교시와 요일 할당
for i in range(5):
    for j in range(5):
        matrix[i, j] = 요일[j] + f' - {교시[i]}'

# 행렬 출력
for i in range(5):
    for j in range(5):
        print(matrix[i, j], end='\t')
    print()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import font_manager
f_path = "C:/windows/Fonts/malgun.ttf"
font_manager.FontProperties(fname=f_path).get_name()
rc('font', family='Malgun Gothic')

# 교시와 요일 목록
교시 = ['1교시', '2교시', '3교시', '4교시', '5교시']
요일 = ['월요일', '화요일', '수요일', '목요일', '금요일']

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=object)

# # 행과 열에 교시와 요일 할당
# for i in range(5):
#     for j in range(5):
#         matrix[i, j] = 요일[j] + f' - {교시[i]}'

# 히트맵을 사용하여 시각화
fig, ax = plt.subplots()
cax = ax.matshow(np.zeros((5, 5), dtype=int), cmap='coolwarm')

# 축과 레이블 설정
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xticklabels(요일)
ax.set_yticklabels(교시)

# # 히트맵에 텍스트 표시
# for i in range(5):
#     for j in range(5):
#         ax.text(j, i, matrix[i, j], ha='center', va='center', color='black')

plt.show()


# In[16]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 각 행마다 무작위로 'A' 배치
for i in range(2):
    j = np.random.randint(4)  # 0부터 4까지의 무작위 열 선택
    matrix[i, j] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[22]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

matrix.fill('X')

def is_valid_placement(matrix, row, col):
    # 주변 8개의 칸 검사 (상하좌우, 대각선)
    neighbors = [
        (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
        (row, col - 1), (row, col + 1),
        (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)
    ]

    for r, c in neighbors:
        if 0 <= r < 5 and 0 <= c < 5 and matrix[r, c] == 'A':
            return False

    return True

for _ in range(2):
    placed = False
    while not placed:
        row = np.random.randint(5)
        col = np.random.randint(5)

        if matrix[row, col] != 'A' and is_valid_placement(matrix, row, col):
            matrix[row, col] = 'A'
            placed = True

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[23]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# "A"를 1열에 무작위로 배치
col1 = np.random.randint(5)
matrix[:, col1] = 'A'

# "A"가 1열에 있으면 3열과 4열에 배치
if col1 == 0:
    matrix[:, 2] = 'A'
    matrix[:, 3] = 'A'
elif col1 == 2:
    matrix[:, 0] = 'A'
    matrix[:, 4] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[25]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 2개의 행 선택
rows = np.random.choice(5, 2, replace=False)

# 첫 번째 "A" 배치
col1 = np.random.randint(5)
matrix[rows[0], col1] = 'A'

# 두 번째 "A" 배치
# 두 번째 "A"를 배치할 때, 같은 열에 있지 않고, 주어진 조건에 따라 1열 또는 5열 중 하나에 배치
for col2 in [0, 4]:
    if col2 != col1:
        matrix[rows[1], col2] = 'A'
        break

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[26]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 첫 번째 "A"를 1열 또는 2열에 배치
col1 = np.random.choice([0, 1])
matrix[:, col1] = 'A'

# 두 번째 "A" 배치
# 두 번째 "A"를 배치할 때, 첫 번째 "A"로부터 2칸 이상 떨어진 열에 배치
# (이때 열이 0 또는 4를 벗어나지 않도록 조건 추가)
valid_columns = [col for col in [2, 3, 4] if abs(col - col1) >= 2 and col != 0 and col != 4]
col2 = np.random.choice(valid_columns)
matrix[:, col2] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[31]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 2개의 열 선택
cols = np.random.choice(5, 2, replace=False)

# 첫 번째 "A" 배치
col1 = cols[0]
matrix[:, col1] = 'A'

# 두 번째 "A" 배치
col2 = cols[1]
# 두 번째 "A"를 배치할 때, 같은 열에 있지 않고, 주어진 조건에 따라 차이가 2 이상 나도록 배치
if abs(col1 - col2) < 2:
    if col2 == 0:
        col2 = 3
    elif col2 == 4:
        col2 = 1
    else:
        col2 = col1 - 2 if np.random.rand() < 0.5 else col1 + 2
matrix[:, col2] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[32]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 2개의 위치 선택
locations = np.random.choice(25, 2, replace=False)
for loc in locations:
    row = loc // 5
    col = loc % 5
    matrix[row, col] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[38]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 2개의 위치 선택
locations = np.random.choice(25, 2, replace=False)

# 첫 번째 "A" 배치
row1, col1 = divmod(locations[0], 5)
matrix[row1, col1] = 'A'

# 두 번째 "A" 배치
row2, col2 = divmod(locations[1], 5)
# 두 번째 "A"를 배치할 때, 첫 번째 "A"와의 행 또는 열 차이가 2 이상이 되도록 배치
while abs(row1 - row2) < 2 and abs(col1 - col2) < 2:
    locations = np.random.choice(25, 2, replace=False)
    row2, col2 = divmod(locations[1], 5)

matrix[row2, col2] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[46]:


import numpy as np

# 5행 5열의 행렬 생성
matrix = np.empty((5, 5), dtype=str)

# 초기화: 모든 요소를 'X'로 채움
matrix.fill('X')

# 무작위로 2개의 위치 선택
locations = np.random.choice(25, 2, replace=False)

# 무작위로 두 위치를 선정하여 첫 번째 "A" 배치
row1, col1 = divmod(locations[0], 5)
matrix[row1, col1] = 'A'

# 두 번째 "A" 배치
row2, col2 = divmod(locations[1], 5)

# 두 번째 "A"를 배치할 때, 같은 열에 위치하지 않도록 배치
while col2 == col1:
    locations = np.random.choice(25, 2, replace=False)
    row2, col2 = divmod(locations[1], 5)

# 두 번째 "A"를 배치할 때, 첫 번째 "A"와의 행 또는 열 차이가 2 이상이 되도록 배치
while abs(row1 - row2) < 2 and abs(col1 - col2) < 2:
    locations = np.random.choice(25, 2, replace=False)
    row2, col2 = divmod(locations[1], 5)

matrix[row2, col2] = 'A'

# 행렬 출력
for row in matrix:
    print(' '.join(row))


# In[ ]:





# In[ ]:




