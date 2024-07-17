print("WELCOME TO PYTHON CALCULATOR")
num1 = int(input("Enter your first number: "))
num2 = int(input("Enter your second number: "))
operator = int(input("Enter number of calculation type :(\n1.Addition\n2.Subtraction\n3.Multiplication\n4.Division\n5.Sqaure of first number\n6.Sqaure root of\n7.Sqaure of second number\n8.Sqaure root of second number\n"))
match operator:
    case 1:
        print(f"Addition is: {num1} + {num2} = {num1 + num2}")
    case 2:
        print(f"Subtraction is: {num1} - {num2} = {num1 - num2}")
    case 3:
        print(f"Multiplication is: {num1} * {num2} = {num1 * num2}")
    case 4:
        print(f"Division is: {num1} / {num2} = {num1 / num2}")
    case 5:
        print(f"Sqaure of first number is: {num1} = {num1**2}")
    case 6:
        print(f"Sqaure root of first number is: {num1} = {num1**0.5}")
    case 7:
        print(f"Sqaure of second number is: {num2} = {num2**2}")
    case 8:
        print(f"Sqaure root of second number is: {num2} = {num2**0.5}")
    case _:
        print("INVALID INPUT\nYOU ARE EXIT")