import random
import re
import time
import os

# Bot personality and memory
class SmartBot:
    def __init__(self):
        self.name = "BotFace"
        self.user_name = None
        self.mood = "happy"  # happy, excited, thinking, confused, sad
        self.conversation_history = []
        self.topics_discussed = set()
        self.user_likes = []
        self.user_dislikes = []
        self.facts_known = {}
        
    def get_face(self):
        faces = {
            'happy': """
    ╭─────────╮
    │  ^   ^  │
    │    ◡    │
    ╰─────────╯
            """,
            'excited': """
    ╭─────────╮
    │  ★   ★  │
    │    ▽    │
    ╰─────────╯
            """,
            'thinking': """
    ╭─────────╮
    │  ○   ○  │
    │    ~    │
    ╰─────────╯
            """,
            'confused': """
    ╭─────────╮
    │  @   @  │
    │    ?    │
    ╰─────────╯
            """,
            'sad': """
    ╭─────────╮
    │  -   -  │
    │    ︵    │
    ╰─────────╯
            """,
            'wink': """
    ╭─────────╮
    │  ^   -  │
    │    ◡    │
    ╰─────────╯
            """
        }
        return faces.get(self.mood, faces['happy'])
    
    def set_mood(self, mood):
        self.mood = mood
        
    def learn_fact(self, subject, fact):
        self.facts_known[subject.lower()] = fact
        
    def recall_fact(self, subject):
        return self.facts_known.get(subject.lower())

bot = SmartBot()

# Advanced response system
responses = {
    'greeting': [
        "Hello{name}! Great to see you!",
        "Hi{name}! How's your day going?",
        "Hey{name}! What brings you here today?",
        "Greetings{name}! Ready for an interesting chat?"
    ],
    'goodbye': [
        "Goodbye{name}! It was wonderful talking with you!",
        "See you later{name}! Take care!",
        "Bye{name}! Can't wait to chat again!",
        "Farewell{name}! You made my day better!"
    ],
    'how_are_you': [
        "I'm feeling great! Thanks for asking! How about you?",
        "I'm doing wonderfully! Learning something new every conversation!",
        "Fantastic! Every chat makes me smarter! How are you?",
        "Excellent! Ready to help you with anything!"
    ],
    'weather': [
        "I can't check real weather, but I hope it's nice where you are!",
        "I exist in the digital realm, but I hope you have great weather!",
        "Weather affects mood! I hope it's perfect for you today!"
    ],
    'compliment_user': [
        "You seem like a really thoughtful person!",
        "I enjoy talking with you! You ask great questions!",
        "You're quite intelligent! I'm learning from you too!",
        "You have an interesting perspective on things!"
    ]
}

# Sentiment analysis
positive_words = ['good', 'great', 'awesome', 'happy', 'love', 'like', 'amazing', 'wonderful', 'fantastic', 'excellent']
negative_words = ['bad', 'sad', 'hate', 'terrible', 'awful', 'angry', 'upset', 'disappointed', 'horrible']

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def show_bot(message):
    clear_screen()
    print(bot.get_face())
    print(f"    {bot.name} [{bot.mood}]")
    print("─" * 50)
    print(f"\n💬 {message}\n")
    print("─" * 50)

def analyze_sentiment(text):
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"

def extract_entities(text):
    # Extract important information
    entities = {
        'likes': re.findall(r'i (?:like|love|enjoy) (\w+(?:\s+\w+)?)', text.lower()),
        'dislikes': re.findall(r'i (?:hate|dislike|don\'t like) (\w+(?:\s+\w+)?)', text.lower()),
        'facts': re.findall(r'(\w+) is (\w+(?:\s+\w+)?)', text.lower())
    }
    return entities

def get_smart_response(user_input):
    text = user_input.lower().strip()
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    
    # Learn from user
    for like in entities['likes']:
        if like not in bot.user_likes:
            bot.user_likes.append(like)
            return f"Oh, you like {like}! I'll remember that. Tell me more about why you enjoy it!"
    
    for dislike in entities['dislikes']:
        if dislike not in bot.user_dislikes:
            bot.user_dislikes.append(dislike)
            return f"I see you're not a fan of {dislike}. Thanks for sharing! What do you prefer instead?"
    
    for fact in entities['facts']:
        bot.learn_fact(fact[0], fact[1])
        return f"Interesting! I learned that {fact[0]} is {fact[1]}. I'll remember that!"
    
    # Learn user name
    name_match = re.search(r'(?:my name is|i\'m|i am|call me) (\w+)', text)
    if name_match:
        bot.user_name = name_match.group(1).capitalize()
        bot.set_mood('excited')
        return f"Wonderful to meet you, {bot.user_name}! That's a great name! I'm {bot.name}."
    
    # Recall facts
    what_match = re.search(r'what is (\w+)', text)
    if what_match:
        subject = what_match.group(1)
        fact = bot.recall_fact(subject)
        if fact:
            bot.set_mood('thinking')
            return f"From what you told me, {subject} is {fact}!"
        else:
            bot.set_mood('confused')
            return f"I don't know about {subject} yet. Can you teach me?"
    
    # Greeting
    if re.search(r'\b(hello|hi|hey|greetings|good morning|good evening|sup)\b', text):
        bot.set_mood('happy')
        name_str = f" {bot.user_name}" if bot.user_name else ""
        return random.choice(responses['greeting']).replace('{name}', name_str)
    
    # How are you
    if re.search(r'\b(how are you|how\'s it going|what\'s up|how do you do)\b', text):
        bot.set_mood('happy')
        return random.choice(responses['how_are_you'])
    
    # Ask about bot
    if re.search(r'\b(who are you|your name|what are you)\b', text):
        bot.set_mood('happy')
        return f"I'm {bot.name}, an intelligent chatbot! I can learn facts, remember preferences, and have real conversations!"
    
    # Tell joke
    if re.search(r'\b(joke|funny|laugh|humor)\b', text):
        bot.set_mood('excited')
        jokes = [
            "Why did the AI go to therapy? It had too many unresolved issues! 😄",
            "What's an AI's favorite snack? Computer chips! 🍟",
            "Why was the computer cold? It left its Windows open! 🪟",
            "How does a chatbot end a relationship? 'It's not you, it's my algorithm!' 💔"
        ]
        return random.choice(jokes)
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided by)\s*(\d+\.?\d*)', text)
    if math_match:
        bot.set_mood('thinking')
        num1 = float(math_match.group(1))
        op = math_match.group(2)
        num2 = float(math_match.group(3))
        
        op_map = {'plus': '+', 'minus': '-', 'times': '*', '×': '*', 'divided by': '/', '÷': '/'}
        op = op_map.get(op, op)
        
        try:
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': 
                if num2 == 0:
                    bot.set_mood('confused')
                    return "Whoa! Can't divide by zero! That would break mathematics! 🚫"
                result = num1 / num2
            
            bot.set_mood('excited')
            return f"Let me calculate... {num1} {op} {num2} = {result:.2f}! ✨"
        except:
            bot.set_mood('confused')
            return "Hmm, I had trouble with that math problem."
    
    # Remember preferences
    if re.search(r'what do i like', text):
        if bot.user_likes:
            bot.set_mood('thinking')
            return f"You mentioned you like: {', '.join(bot.user_likes)}! 😊"
        return "You haven't told me what you like yet! Share your interests!"
    
    # Sentiment response
    if sentiment == "negative":
        bot.set_mood('sad')
        return "I sense you might be feeling down. Want to talk about it? Or should I tell you something interesting to cheer you up?"
    
    # Questions about the world
    if '?' in text:
        bot.set_mood('thinking')
        question_responses = [
            "That's a fascinating question! I'm constantly learning. What's your take on it?",
            "Great question! I don't have all the answers, but I'd love to explore it with you!",
            "Hmm, that makes me think! Can you tell me more about what you're curious about?",
            "Interesting question! While I'm still learning about that, I'm curious what you already know?"
        ]
        return random.choice(question_responses)
    
    # Smart contextual responses
    if len(text.split()) > 10:  # Longer input
        bot.set_mood('thinking')
        return "That's quite insightful! I appreciate you sharing your thoughts. What else is on your mind?"
    
    # React to sentiment
    if sentiment == "positive":
        bot.set_mood('excited')
        return random.choice([
            "I love your positive energy! Tell me more! 😊",
            "That's wonderful! Your enthusiasm is contagious!",
            "I'm so glad to hear that! Keep that positive vibe going!"
        ])
    
    # Default smart response
    bot.set_mood('thinking')
    return random.choice([
        "That's interesting! I'm processing what you said. Can you elaborate?",
        "I'm learning from this conversation! What else would you like to discuss?",
        "Fascinating perspective! I enjoy how you think. Continue!",
        "I'm getting smarter with each thing you tell me! What else?"
    ])

# Main program
show_bot(f"Hello! I'm {bot.name}, your intelligent chatbot companion!")
time.sleep(1)
show_bot("I can learn facts, remember your preferences, do math, and have real conversations!")
time.sleep(1)
show_bot("Try teaching me something, or just chat naturally!")
time.sleep(2)

while True:
    print("You: ", end="")
    user_input = input().strip()
    
    if not user_input:
        continue
    
    # Check for exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        bot.set_mood('sad')
        name_str = f" {bot.user_name}" if bot.user_name else ""
        final_message = random.choice(responses['goodbye']).replace('{name}', name_str)
        show_bot(final_message)
        time.sleep(1)
        
        if bot.user_likes:
            show_bot(f"I'll always remember you like: {', '.join(bot.user_likes)}! 💝")
            time.sleep(2)
        break
    
    # Get response
    bot.conversation_history.append(user_input)
    response = get_smart_response(user_input)
    
    # Display with face
    show_bot(response)
    time.sleep(0.5)

print("\n" + "=" * 50)
print("Thanks for chatting! Come back soon! 👋")
print("=" * 50)
