#!/usr/bin/env python3
"""FizzBuzz CLI - prints numbers from 1 to N with FizzBuzz rules."""

import sys


def main():
    """Read integer N and print numbers from 1 to N using FizzBuzz rules."""
    if len(sys.argv) != 2:
        print("Usage: python3 fizzbuzz.py <N>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer")
        sys.exit(1)
    
    if n <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)
    
    for i in range(1, n + 1):
        result = ""
        if i % 3 == 0:
            result += "Fizz"
        if i % 5 == 0:
            result += "Buzz"
        
        if result:
            print(result)
        else:
            print(i)


if __name__ == "__main__":
    main()
