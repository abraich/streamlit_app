# defie a function that retruns all prime numbers up to a given number
def prime_numbers(n):
    """
    A function that returns all prime numbers up to a given number
    """
    prime_numbers = []
    for num in range(2, n+1):
        for i in range(2, num):
            if (num % i) == 0:
                break
        else:
            prime_numbers.append(num)
    return prime_numbers

# test the function
print(prime_numbers(10))
