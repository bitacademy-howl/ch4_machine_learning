import numpy as np

xy = np.loadtxt("test-score.csv", delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))

print(x_data[:, 0])

# 요거는 step 에 정수만 들어감, range 객체로 생성됨
nums = range(0, 5)
# range 의 속성과 멤버메서드들...
print(nums.index(2))
print(nums.step, nums.start, nums.stop)

print(type(nums))
print(nums)

print(nums[2:4])
print(nums[:-1])

# 재배열?혹은 재지정
nums[2:4] = [8, 9]
print(nums)

# 요거는 step 에 float 가능 하지만, numpy.ndarray 객체를 반환함
nums = np.arange(0, 5, 0.1)
print(type(nums))