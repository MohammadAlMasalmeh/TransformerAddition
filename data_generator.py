import random

def format_num(num, width):
    sign = "-" if num < 0 else "+"
    return sign + str(abs(num)).zfill(width)

def make_data_point():
    num1 = random.randint(-999, 999)
    num2 = random.randint(-999, 999)
    total = num1 + num2
    output = format_num(num1, 3) + " " + format_num(num2, 3) + " = " + format_num(total, 4)

    return output

def generate_dataset(n):
    data = []
    for i in range(n):
        data.append(make_data_point())
    return data


print(generate_dataset(10))