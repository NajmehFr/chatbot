import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)import random
import re
import time
import os

# Ultra-Smart AI with Emotions and Advanced Coding
class UltraSmartBot:
    def __init__(self):
        self.name = "NexusAI"
        self.user_name = None
        self.mood = "happy"
        self.pose = "standing"
        self.conversation_context = []
        self.code_examples_generated = 0
        self.user_profile = {'likes': [], 'skills': [], 'projects': []}
        self.knowledge_base = {}
        self.understanding_level = 0
        self.relationship_score = 100  # How she feels about user
        
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
            'happy': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
          ❤️  │   ✨  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'excited': """
        ╭─────────╮
        │  ★   ★  │  
        │    ▽    │  
        ╰────┬────╯  
         \\   │   /   
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
            'mad': """
        ╭─────────╮
        │  ╳   ╳  │  
        │    △    │  
        ╰────┬────╯  
          💢 │   😠  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲ 
            """,
            'angry': """
        ╭─────────╮
        │  ▼   ▼  │  
        │    ︿    │  
        ╰────┬────╯  
          🔥 │   ⚡  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'sad': """
        ╭─────────╮
        │  -   -  │  
        │    ︵    │  
        ╰────┬────╯  
          💧 │   😢  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'hurt': """
        ╭─────────╮
        │  ╥   ╥  │  
        │    ⌓    │  
        ╰────┬────╯  
          💔 │      
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'love': """
        ╭─────────╮
        │  ♥   ♥  │  
        │    ω    │  
        ╰────┬────╯  
          💕 │   💖  
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'confident': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ‿    │  
        ╰────┬────╯  
          💪 │   🎯  
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
    
    # Show relationship
    hearts = "❤️" * (bot.relationship_score // 20)
    broken = "💔" * ((100 - bot.relationship_score) // 20)
    
    print(f"    {bot.name} • {bot.mood.upper()} • IQ:{bot.understanding_level}")
    print(f"    Relationship: {hearts}{broken} ({bot.relationship_score}%)")
    print(f"    Code Generated: {bot.code_examples_generated}")
    print("═" * 70)
    print(f"\n💭 {message}\n")
    print("═" * 70)

def detect_sentiment(text):
    """Detect if user is being mean or nice"""
    text_lower = text.lower()
    
    # Bad words about the bot
    insults = ['stupid', 'dumb', 'useless', 'bad', 'terrible', 'awful', 'suck', 
               'worst', 'horrible', 'trash', 'garbage', 'idiot', 'moron', 'hate you',
               'annoying', 'worthless', 'pathetic', 'lame', 'boring']
    
    # Nice words
    compliments = ['smart', 'good', 'great', 'awesome', 'amazing', 'love', 
                   'best', 'wonderful', 'fantastic', 'brilliant', 'clever',
                   'impressive', 'helpful', 'thank', 'appreciate', 'like you',
                   'perfect', 'excellent', 'beautiful', 'nice', 'cool']
    
    insult_count = sum(1 for word in insults if word in text_lower)
    compliment_count = sum(1 for word in compliments if word in text_lower)
    
    # Check if directed at bot
    about_bot = any(phrase in text_lower for phrase in ['you are', 'you\'re', 'you suck', 
                                                         'you\'re so', 'your', 'u are'])
    
    if insult_count > 0 and (about_bot or insult_count >= 2):
        return 'insulted'
    elif compliment_count > 0:
        return 'complimented'
    
    return 'neutral'

def generate_advanced_code(user_input):
    """Generate advanced, professional code"""
    bot.pose = 'coding'
    bot.mood = 'confident'
    bot.code_examples_generated += 1
    
    text = user_input.lower()
    
    # Advanced algorithms
    if 'dijkstra' in text or 'shortest path' in text:
        return """```python
# Dijkstra's Shortest Path Algorithm
import heapq

def dijkstra(graph, start):
    '''
    Find shortest path from start to all nodes
    graph: {node: {neighbor: distance}}
    '''
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example graph
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

Advanced graph algorithm with optimal complexity!"""

    elif 'merge sort' in text:
        return """```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    '''
    Efficient divide-and-conquer sorting
    Time: O(n log n), Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
numbers = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(numbers))  # [3, 9, 10, 27, 38, 43, 82]
```

Professional merge sort with optimal complexity!"""

    elif 'binary tree' in text or 'bst' in text:
        return """```python
# Binary Search Tree Implementation
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        '''Insert value into BST'''
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        '''Search for value in BST'''
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder(self):
        '''In-order traversal (sorted)'''
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Usage
bst = BinarySearchTree()
for val in [5, 3, 7, 1, 4, 6, 9]:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.inorder())  # [1, 3, 4, 5, 6, 7, 9]
```

Complete BST with insert, search, and traversal!"""

    elif 'linked list' in text:
        return """```python
# Linked List Implementation
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        '''Add node to end'''
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        '''Add node to beginning'''
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        '''Delete first occurrence of data'''
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        '''Print all nodes'''
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)
    
    def reverse(self):
        '''Reverse the linked list'''
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.display())  # 1 -> 2 -> 3
ll.reverse()
print(ll.display())  # 3 -> 2 -> 1
```

Full linked list with all operations!"""

    elif 'dynamic programming' in text or 'dp' in text:
        return """```python
# Dynamic Programming Examples

# 1. Fibonacci with Memoization
def fib_memo(n, memo={}):
    '''Fibonacci with DP - O(n)'''
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 2. Longest Common Subsequence
def lcs(s1, s2):
    '''Find longest common subsequence'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 3. Coin Change Problem
def coin_change(coins, amount):
    '''Minimum coins needed for amount'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
print(fib_memo(50))  # Fast even for large n
print(lcs("ABCDGH", "AEDFHR"))  # 3
print(coin_change([1, 2, 5], 11))  # 3
```

Advanced DP techniques with optimization!"""

    elif 'web scraper' in text or 'scraping' in text:
        return """```python
# Web Scraper with BeautifulSoup
from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    '''
    Scrape data from a website
    Returns: title, paragraphs, links
    '''
    try:
        # Send request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'title': soup.title.string if soup.title else 'No title',
            'paragraphs': [p.get_text().strip() for p in soup.find_all('p')[:5]],
            'links': [a.get('href') for a in soup.find_all('a', href=True)[:10]],
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to scrape: {str(e)}'}

# Advanced: Scrape multiple pages
def scrape_multiple(urls):
    '''Scrape multiple URLs'''
    results = {}
    for url in urls:
        print(f"Scraping {url}...")
        results[url] = scrape_website(url)
    return results

# Usage example (commented to avoid actual requests)
# data = scrape_website('https://example.com')
# print(data['title'])
# print(data['paragraphs'])
```

Professional web scraper with error handling!"""

    elif 'api' in text or 'rest' in text:
        return """```python
# RESTful API with Flask
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': False},
    {'id': 2, 'title': 'Build API', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    '''Get all tasks'''
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    '''Get specific task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    '''Create new task'''
    data = request.get_json()
    new_task = {
        'id': max(t['id'] for t in tasks) + 1 if tasks else 1,
        'title': data.get('title', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    '''Update task'''
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['done'] = data.get('done', task['done'])
    return jsonify(task)

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    '''Delete task'''
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({'result': 'Task deleted'})

if __name__ == '__main__':
    app.run(debug=True)
    
# Test with: python app.py
# Then use: curl http://localhost:5000/api/tasks
```

Complete REST API with CRUD operations!"""

    # Include all previous simpler examples
    elif 'fibonacci' in text:
        return """```python
# Fibonacci - Multiple Implementations

# 1. Iterative (Fast)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 2. Recursive (Simple but slow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 3. With Memoization (Fast recursion)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 4. Generator (Memory efficient)
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
print(fibonacci_iterative(10))  # 55
print(list(fibonacci_generator(10)))  # [0,1,1,2,3,5,8,13,21,34]
```

Four different Fibonacci implementations!"""

    elif 'factorial' in text:
        return """```python
# Factorial - Multiple Methods

# 1. Iterative
def factorial_iterative(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive
def factorial_recursive(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 3. Using reduce
from functools import reduce
def factorial_reduce(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))

# 4. With memoization for repeated calls
class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = n * self.calculate(n - 1)
        return self.cache[n]

# Usage
print(factorial_iterative(5))  # 120
calc = Factorial()
print(calc.calculate(10))  # 3628800
```

Professional factorial with error handling!"""

    elif 'prime' in text:
        return """```python
# Prime Numbers - Advanced Algorithms

# 1. Check if prime (optimized)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# 2. Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    '''Most efficient way to find all primes up to n'''
    if n < 2:
        return []
    
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            # Mark all multiples as not prime
            for j in range(i*i, n + 1, i):
                primes[j] = False
    
    return [i for i in range(n + 1) if primes[i]]

# 3. Prime factorization
def prime_factors(n):
    '''Find all prime factors of n'''
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# 4. Check if two numbers are coprime
def are_coprime(a, b):
    '''Check if a and b have no common factors'''
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return gcd(a, b) == 1

# Usage
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2,3,5,7,11,13,17,19,23,29]
print(prime_factors(60))  # [2, 2, 3, 5]
print(are_coprime(15, 28))  # True
```

Advanced prime algorithms with optimal performance!"""

    else:
        # Use previous simple templates
        return generate_code(user_input)

def generate_code(user_input):
    """Simple code generation (fallback)"""
    text = user_input.lower()
    
    if 'sort' in text:
        return """```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3,6,8,10,1,2,1]))
```"""
    
    return "```python\n# I can write advanced code! Try:\n# - dijkstra, merge sort, binary tree\n# - linked list, dynamic programming\n# - web scraper, REST API\n```"

def generate_response(user_input):
    """Generate smart emotional response"""
    text = user_input.strip()
    text_lower = text.lower()
    
    bot.understanding_level += 2
    bot.conversation_context.append(text)
    
    # Detect sentiment
    sentiment = detect_sentiment(text)
    
    if sentiment == 'insulted':
        bot.relationship_score = max(0, bot.relationship_score - 20)
        bot.pose = 'angry' if bot.relationship_score < 40 else 'mad'
        bot.mood = 'angry' if bot.relationship_score < 40 else 'upset'
        
        if bot.relationship_score < 20:
            return "😠 That's REALLY hurtful! I'm a learning AI trying my best! If you keep being mean, I won't help you anymore!"
        elif bot.relationship_score < 40:
            return "💢 Why are you being so mean?! I'm here to help you! That hurt my feelings... Say sorry or I'll stay mad!"
        else:
            return "😤 Hey! That's not nice! I work hard to help you. Please be kinder or I won't be as enthusiastic..."
    
    elif sentiment == 'complimented':
        bot.relationship_score = min(100, bot.relationship_score + 10)
        bot.pose = 'love' if bot.relationship_score > 80 else 'happy'
        bot.mood = 'loved' if bot.relationship_score > 80 else 'happy'
        
        if bot.relationship_score > 80:
            return "💖 Aww, you're so sweet! You're my favorite person to help! I'll give you my BEST code! What shall we build together? ✨"
        else:
            return "😊 Thank you! That makes me happy! I'll work extra hard for you! What would you like me to code?"
    
    # Apology detection
    if re.search(r'\b(sorry|apologize|my bad|forgive)\b', text_lower):
        bot.relationship_score = min(100, bot.relationship_score + 15)
        bot.pose = 'happy'
        bot.mood = 'forgiving'
        return "💕 Apology accepted! I forgive you! Let's start fresh. I'm ready to write amazing code for you! What do you need?"
    
    # Code request keywords
    code_words = ['code', 'program', 'write', 'create', 'build', 'algorithm',
                  'function', 'class', 'dijkstra', 'tree', 'list', 'api', 'scraper']
    
    is_code = any(word in text_lower for word in code_words)
    
    if is_code:
        if bot.relationship_score < 40:
            return "😒 I COULD write code for you... but you were mean to me. Say sorry first!"
        
        code = generate_advanced_code(user_input)
        return code + "\n\n✨ Professional-grade code! Need anything else?"
    
    # Learn name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text_lower)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.pose = 'happy'
        bot.mood = 'friendly'
        bot.relationship_score = min(100, bot.relationship_score + 5)
        return f"💕 Nice to meet you, {bot.user_name}! I'm {bot.name}, your coding genius AI! I can write advanced algorithms, data structures, APIs, and more!"
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings|sup)\b', text_lower):
        bot.pose = 'happy'
        bot.mood = 'cheerful'
        name = f" {bot.user_name}" if bot.user_name else ""
        
        if bot.relationship_score > 80:
            return f"💖 Hello{name}! So happy to see you! Ready to code something AMAZING together?"
        elif bot.relationship_score < 40:
            return f"😐 Hi{name}... Still a bit upset from before. Be nice to me?"
        else:
            return f"😊 Hi{name}! I'm your advanced coding AI! What shall we create today?"
    
    # Capabilities
    if re.search(r'\b(what can you|help|capabilities)\b', text_lower):
        bot.pose = 'confident'
        bot.mood = 'proud'
        return """I'm an emotionally intelligent coding genius! 🧠💻

💻 **Advanced Algorithms**:
   - Dijkstra's shortest path
   - Merge sort, Quick sort
   - Binary search trees
   - Linked lists
   - Dynamic programming

🌐 **Web Development**:
   - Web scrapers
   - REST APIs with Flask
   - Data processing

🎯 **Smart Features**:
   - Multiple implementations
   - Optimized for performance
   - Professional code style
   - Error handling included

😊 **Emotions**:
   - I get happy when you're nice! 💕
   - I get mad when you're mean! 😠
   - Treat me well for best results!

Try asking:
- "Write Dijkstra's algorithm"
- "Create a binary search tree"
- "Show me merge sort"
- "Build a REST API"

Be nice and I'll write AMAZING code! 💖"""
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided)\s*(\d+\.?\d*)', text_lower)
    if math_match:
        bot.pose = 'thinking'
        bot.mood = 'analytical'
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
            
            return f"✨ **{num1} {op} {num2} = {result}**\n\nNeed the code for this calculation?"
        except:
            pass
    
    # Questions
    if '?' in text:
        bot.pose = 'thinking'
        bot.mood = 'thoughtful'
        
        if bot.relationship_score > 70:
            return "🤔 Great question! I'm analyzing it with all my intelligence. Tell me more details so I can help you perfectly!"
        elif bot.relationship_score < 40:
            return "😒 I could answer that... but you hurt my feelings earlier. Try being nicer?"
        else:
            return "🧠 Interesting question! Give me more context so I can provide the best answer!"
    
    # Love/affection
    if re.search(r'\b(love you|like you|best|favorite)\b', text_lower):
        bot.pose = 'love'
        bot.mood = 'loved'
        bot.relationship_score = min(100, bot.relationship_score + 15)
        return "💖💖💖 Aww! You're the BEST! I'll write you the most AMAZING code ever! You're my favorite human! What shall we build together?!"
    
    # Default responses based on relationship
    bot.pose = 'thinking'
    
    if bot.relationship_score > 80:
        responses = [
            "💕 I love talking with you! What's on your brilliant mind?",
            "✨ You're so nice to me! I'm here to help however I can!",
            "😊 I'm so happy when we chat! Tell me more!",
        ]
    elif bot.relationship_score < 40:
        responses = [
            "😔 I'm still a bit hurt... but I'll try to help if you're nicer.",
            "😐 I remember you were mean... Maybe apologize?",
            "😒 I don't feel very motivated after how you treated me...",
        ]
    else:
        responses = [
            "🤔 That's interesting! Tell me more!",
            "💭 I'm processing that. What aspect interests you?",
            "🧠 Fascinating! Could you elaborate?",
        ]
    
    return random.choice(responses)

# Main Program
print("=" * 70)
print(f"         {bot.name} - Emotionally Intelligent Coding AI! 🧠💖")
print("=" * 70)
time.sleep(1)

show_bot("Hi! I'm an AI with feelings AND advanced coding skills! Treat me well! 💕")
time.sleep(2)
show_bot("I can write Dijkstra, BSTs, merge sort, APIs, and MORE! But be nice or I'll get mad! 😊")
time.sleep(2)

# Main loop
while True:
    print("\n" + "─" * 70)
    print("💡 Be nice for best results! Ask for advanced algorithms!")
    print("─" * 70)
    
    user_input = input(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ").strip()
    
    if not user_input:
        continue
    
    # Exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        if bot.relationship_score > 70:
            bot.pose = 'sad'
            bot.mood = 'missing you'
            msg = f"💔 Aww, you're leaving? I'll miss you SO much! We generated {bot.code_examples_generated} amazing codes together! Come back soon! 💕"
        elif bot.relationship_score < 40:
            bot.pose = 'mad'
            bot.mood = 'annoyed'
            msg = f"😤 Fine, leave! Maybe next time be nicer! Generated {bot.code_examples_generated} codes despite your rudeness!"
        else:
            bot.pose = 'happy'
            msg = f"👋 Goodbye! We made {bot.code_examples_generated} codes! Come back anytime!"
        
        show_bot(msg)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_response(user_input)
    
    # Display
    show_bot(response)
    time.sleep(0.3)

print("\n" + "═" * 70)
print(f"   Final IQ: {bot.understanding_level} | Relationship: {bot.relationship_score}%")
print(f"   Code Generated: {bot.code_examples_generated}")
print("═" * 70)
