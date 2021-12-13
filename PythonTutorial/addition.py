def add(num1, num2):
	sum = num1 + num2
	return sum

if __name__ == "__main__":
	a = 1
	b = 2
	c = add(a, b)
	print('a = {}, b = {}, a + b = {}'.format(a, b, c))

	a = 2
	b = 3
	c = add(a, b)
	print('a = {}, b = {}, a + b = {}'.format(a, b, c))