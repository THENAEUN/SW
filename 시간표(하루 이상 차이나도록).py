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
while True:
    row2, col2 = divmod(locations[1], 5)
    # 두 번째 "A"를 배치할 때, 같은 열에 위치하지 않도록 보정
    if col2 == col1:
        locations = np.random.choice(25, 2, replace=False)
    else:
        break

# 두 번째 "A"를 배치할 때, 첫 번째 "A"와의 열 차이가 2 이상이 되도록 배치
while abs(col1 - col2) < 2:
    locations = np.random.choice(25, 2, replace=False)
    row2, col2 = divmod(locations[1], 5)

matrix[row2, col2] = 'A'

# 행렬 출력
for i in range(5):
    for j in range(5):
        print(matrix[i, j], end='\t')
    print()
