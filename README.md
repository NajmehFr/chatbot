import random
import re
import time
import os

# Ultra-Smart AI with Code Generation
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "analytical"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        
    def get_body(self):
        bodies = {
            'standing': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ▽    │  
        ╰────┬────╯  
             │   💡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'thinking': """
        ╭─────────╮
        │  ●   ●  │  
        │    ~    │  
        ╰────┬────╯  
          🧠 │   💭  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'coding': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ▭    │  
        ╰────┬────╯  
         </> │   💻  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'teaching': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          📚 │   ✏️   
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
        }
        return bodies.get(self.pose, bodies['standing'])

bot = UltraSmartBot()

def clear_screen():
    try:
        os.system('clear' if os.name == 'posix' else 'cls')
    except:
        print("\n" * 50)

def show_bot(message):
    clear_screen()
    print(bot.get_body())
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level} • Code:{bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def generate_code(user_input):
    """Generate code based on request"""
    bot.pose = 'coding'
    bot.mood = 'creative'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Detect specific requests
    if 'fibonacci' in text:
        return """```python
# Fibonacci sequence generator
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Usage
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

Generates the first n Fibonacci numbers!"""

    elif 'factorial' in text:
        return """```python
# Factorial calculator
def factorial(n):
    if n < 0:
        return "Undefined"
    elif n <= 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Recursive version
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Usage
print(factorial(5))  # 120
```

Two approaches: iterative and recursive!"""

    elif 'prime' in text:
        return """```python
# Check if number is prime
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Get all primes up to n
def primes_up_to(n):
    return [i for i in range(2, n+1) if is_prime(i)]

# Usage
print(is_prime(17))  # True
print(primes_up_to(20))  # [2, 3, 5, 7, 11, 13, 17, 19]
```

Efficient prime number checking!"""

    elif 'bubble' in text and 'sort' in text:
        return """```python
# Bubble Sort Algorithm
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Usage
numbers = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(numbers))  # [11, 12, 22, 25, 34, 64, 90]
```

Classic bubble sort with optimization!"""

    elif 'sort' in text:
        return """```python
# Quick Sort Algorithm
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Usage
numbers = [64, 34, 25, 12, 22, 11, 90]
print(quick_sort(numbers))  # [11, 12, 22, 25, 34, 64, 90]
```

Fast and elegant quick sort!"""

    elif 'binary' in text and 'search' in text:
        return """```python
# Binary Search Algorithm
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Usage
numbers = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(numbers, 7))  # 3
print(binary_search(numbers, 10))  # -1
```

O(log n) binary search!"""

    elif 'search' in text:
        return """```python
# Linear Search
def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1

# Find all occurrences
def find_all(arr, target):
    return [i for i, x in enumerate(arr) if x == target]

# Usage
numbers = [4, 2, 7, 1, 7, 3, 7]
print(linear_search(numbers, 7))  # 2
print(find_all(numbers, 7))  # [2, 4, 6]
```

Simple linear search!"""

    elif 'class' in text or 'oop' in text:
        return """```python
# Object-Oriented Programming
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.friends = []
    
    def introduce(self):
        return f"Hi! I'm {self.name}, {self.age} years old."
    
    def add_friend(self, friend):
        self.friends.append(friend)
        print(f"{self.name} is friends with {friend.name}!")
    
    def birthday(self):
        self.age += 1
        print(f"Happy birthday {self.name}! Now {self.age}!")

# Usage
alice = Person("Alice", 25)
bob = Person("Bob", 30)
print(alice.introduce())
alice.add_friend(bob)
```

Clean OOP with methods!"""

    elif 'file' in text:
        return """```python
# File Operations
def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found!"

def write_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Written to {filename}")

def append_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content + '\\n')

# Usage
write_file('test.txt', 'Hello World!')
print(read_file('test.txt'))
```

Safe file handling!"""

    elif 'calculator' in text:
        return """```python
# Simple Calculator
def calculator():
    print("Calculator: +, -, *, /, **, %")
    
    while True:
        try:
            num1 = float(input("First number (q to quit): "))
            op = input("Operation: ")
            num2 = float(input("Second number: "))
            
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2 if num2 != 0 else "Error!"
            elif op == '**': result = num1 ** num2
            elif op == '%': result = num1 % num2
            else: result = "Invalid!"
            
            print(f"Result: {result}\\n")
        except:
            break

# calculator()  # Uncomment to run
```

Full calculator with operators!"""

    elif 'game' in text:
        return """```python
# Guessing Game
import random

def guessing_game():
    print("🎮 Guess the Number (1-100)!")
    number = random.randint(1, 100)
    attempts = 0
    
    while attempts < 10:
        try:
            guess = int(input(f"Attempt {attempts+1}: "))
            attempts += 1
            
            if guess < number:
                print("📈 Too low!")
            elif guess > number:
                print("📉 Too high!")
            else:
                print(f"🎉 Won in {attempts} attempts!")
                return
        except:
            print("Enter a number!")
    
    print(f"Game Over! Number was {number}")

# guessing_game()  # Uncomment to play
```

Fun guessing game!"""

    elif 'list' in text or 'array' in text:
        return """```python
# List Operations
# Create lists
numbers = [1, 2, 3, 4, 5]
fruits = ['apple', 'banana', 'orange']

# Add items
numbers.append(6)
numbers.insert(0, 0)

# Remove items
numbers.remove(3)
last = numbers.pop()

# Access
first = numbers[0]
last = numbers[-1]

# Slice
first_three = numbers[:3]
last_two = numbers[-2:]

# Loop
for num in numbers:
    print(num)

# Comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in numbers if x % 2 == 0]

print(numbers)
```

Complete list operations!"""

    elif 'loop' in text:
        return """```python
# Loop Examples

# For loop with range
for i in range(5):
    print(f"Count: {i}")

# For loop with list
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(f"I like {fruit}")

# For loop with enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# While loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Loop with break
for i in range(10):
    if i == 5:
        break
    print(i)

# Loop with continue
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)
```

All types of loops!"""

    else:
        return """```python
# Python Code Template
def process_data(data):
    result = []
    for item in data:
        processed = item * 2
        result.append(processed)
    return result

class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def process(self):
        return [x * 2 for x in self.data]

# Usage
numbers = [1, 2, 3, 4, 5]
print(process_data(numbers))  # [2, 4, 6, 8, 10]
```

Flexible code template!"""

def generate_response(user_input):
    """Generate smart response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'make', 
                  'show', 'example', 'function', 'algorithm', 'how to']
    
    is_code = any(word in text_lower for word in code_words)
    
    # Check if asking for code
    if is_code or 'fibonacci' in text_lower or 'sort' in text_lower or 'search' in text_lower:
        code = generate_code(user_input)
        return code + "\n\nWell-structured code with comments! Need changes?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'teaching'
        return f"Nice to meet you, {bot.user_name}! I'm {bot.name}, your smart AI assistant with coding powers!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'teaching'
        name = f" {bot.user_name}" if bot.user_name else ""
        return f"Hello{name}! I'm {bot.name}, an AI that writes code and solves problems. What can I help you with?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'teaching'
        return """I'm a smart AI that can:

🧠 Deep Reasoning - Solve complex problems
💻 Code Generation - Write Python code
📊 Algorithms - Sorting, searching, more
🎓 Teaching - Explain concepts
🔢 Math - Calculate anything

Try:
- "Write a fibonacci function"
- "Show me a sorting algorithm"
- "Create a calculator"
- "Explain binary search"

What interests you?"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        num1 = float(math_match.group(1))
        op = math_match.group(2)
        num2 = float(math_match.group(3))
        
        ops = {'plus': '+', 'minus': '-', 'times': '*', '×': '*', 'divided': '/', '÷': '/'}
        op = ops.get(op, op)
        
        try:
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2 if num2 != 0 else "undefined"
            
            return f"**{num1} {op} {num2} = {result}**\n\nWant code for this?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        return "That's a great question! I'm analyzing it. Can you be more specific so I can give you the best answer?"
    
    # Default
    bot.pose = 'thinking'
    responses = [
        "Interesting! Tell me more about that.",
        "I'm processing that. What aspect interests you most?",
        "That's thought-provoking! Could you elaborate?",
        "I see! What would you like to explore?",
    ]
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Ultra-Smart AI Online! 🧠💻")
print("=" * 70)
time.sleep(1)

show_bot("I can write code, solve problems, and explain concepts!")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Ask me to write code or solve problems!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        bot.pose = 'teaching'
        msg = f"Goodbye! Generated {bot.code_examples_generated} code examples! IQ: {bot.understanding_level}! 🚀"
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   IQ: {bot.understanding_level} | Code: {bot.code_examples_generated}")
print("═" * 70)
