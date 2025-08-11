def factorial(n):
    """
    Calculates the factorial of n with an artificial O(n^2) runtime.
    This is for demonstration purposes only, as it's an inefficient way to calculate factorial.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")

    if n == 0:
        return 1

    result = 1
    for i in range(1, n + 1):  # Outer loop runs n times
        temp_sum = 0
        for j in range(n):  # Inner loop runs n times
            temp_sum += 1  # Dummy operation to increase complexity
        result *= i
    return result

