# For style 21: recursive/iterative
def factorial_recursive(n):
    if n == 0:
        return 1
    else:
        return n * factorial_recursive(n - 1)

# For style 17: nested if
def check_nested(a, b):
    if a > 0 and b > 0:
        print("Both positive")

# For style 16: if/switch (Python has no switch, so this is for if-elif-else)
def print_day(day):
    if day == 1:
        print("Monday")
    elif day == 2:
        print("Tuesday")
    else:
        print("Other day")

def main():
    # For style 0: identifier name
    my_variable = 10
    my_variable_two = 20

    # For style 2: augmented assignment
    my_variable = my_variable + 5
    my_variable_two += 5

    # For style 1, 11: brackets (not really applicable), for/while
    for i in range(5):
        my_variable += 1

    j = 0
    while j < 3:
        my_variable_two -= 1
        j += 1

    # For style 19: ternary
    max_val = my_variable if my_variable > my_variable_two else my_variable_two

    # For style 14: if exclamation
    if not (my_variable == my_variable_two):
        print("Not equal")

    # For style 18: if/else
    if my_variable > 15:
        print("Greater than 15")
    else:
        print("Not greater than 15")

    # For style 8: declare position (Python is dynamic, but can simulate with first use)
    temp_var = my_variable + my_variable_two
    print(f"Result: {temp_var}")

if __name__ == "__main__":
    main()
