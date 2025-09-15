def main():
    total = 0
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for num in numbers:
        if num % 2 == 0:
            total += num
        elif num == 5:
            total -= num
        else:
            total *= num
    
    counter = 0
    while counter < 5:
        if counter > 2:
            total += counter
        counter += 1
    
    return total

if __name__ == "__main__":
    result = main()
    print(f"Result: {result}")
