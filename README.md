# chatbot
import random
import re
import time
import os
from datetime import datetime

# Advanced AI Bot with deep understanding
class IntelligentBot:
    def __init__(self):
        self.name = "Nexus"
        self.user_name = None
        self.mood = "neutral"
        self.pose = "standing"  # standing, thinking, excited, waving, shrugging
        self.conversation_context = []
        self.user_profile = {
            'likes': [], 'dislikes': [], 'hobbies': [], 'goals': [],
            'location': None, 'age': None, 'occupation': None
        }
        self.knowledge_base = {}
        self.topic_expertise = {
            'science': ['physics', 'chemistry', 'biology', 'astronomy'],
            'tech': ['programming', 'ai', 'computer', 'coding', 'python', 'java'],
            'math': ['algebra', 'geometry', 'calculus', 'arithmetic'],
            'arts': ['music', 'painting', 'drawing', 'poetry', 'writing'],
            'sports': ['football', 'basketball', 'soccer', 'tennis', 'swimming']
        }
        self.last_topic = None
        self.understanding_level = 0  # Gets smarter over time
        
    def get_body(self):
        bodies = {
            'standing': """
        ╭─────────╮
        │  ◉   ◉  │  
        │    ▽    │  
        ╰────┬────╯  
             │       
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
          💭 │       
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
            'waving': """
        ╭─────────╮
        │  ^   ^  │  
        │    ◡    │  
        ╰────┬────╯  
             │   ╱   
        ────┼┼┼      
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'shrugging': """
        ╭─────────╮
        │  ╹   ╹  │  
        │    ︵    │  
        ╰────┬────╯  
          ╱  │  ╲    
        ────┼┼┼────  
             │       
            ╱ ╲      
           ╱   ╲     
          ╱     ╲    
            """,
            'celebrating': """
        ╭─────────╮
        │  ◕   ◕  │  
        │    ▿    │  
        ╰────┬────╯  
         \\   │   /   
          \\┼┼┼/      
             │       
            ╱│╲      
           ╱ │ ╲     
          ╱  │  ╲    
            """
        }
        return bodies.get(self.pose, bodies['standing'])

bot = IntelligentBot()

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def show_bot(message, delay=0.03):
    clear_screen()
    print(bot.get_body())
    print(f"      {bot.name} • {bot.mood.upper()} • LV.{bot.understanding_level}")
    print("═" * 60)
    
    # Animated typing effect
    print("\n💭 ", end="", flush=True)
    for char in message:
        print(char, end="", flush=True)
        time.sleep(delay)
    print("\n")
    print("═" * 60)

def extract_intent(text):
    """Advanced intent recognition"""
    text_lower = text.lower()
    
    intents = {
        'greeting': r'\b(hello|hi|hey|greetings|good morning|good evening|sup|yo)\b',
        'farewell': r'\b(bye|goodbye|see you|farewell|exit|quit|later)\b',
        'question_identity': r'\b(who are you|what are you|your name|tell me about yourself)\b',
        'question_capability': r'\b(what can you do|your abilities|help|capabilities|how do you work)\b',
        'question_feeling': r'\b(how are you|how do you feel|are you okay|you good)\b',
        'sharing_like': r'\bi (like|love|enjoy|adore) (\w+)',
        'sharing_dislike': r'\bi (hate|dislike|can\'t stand) (\w+)',
        'question_why': r'\bwhy\b',
        'question_how': r'\bhow\b',
        'question_what': r'\bwhat\b',
        'question_where': r'\bwhere\b',
        'question_when': r'\bwhen\b',
        'teaching': r'\b(\w+) is (\w+(?:\s+\w+)?)',
        'opinion_request': r'\b(what do you think|your opinion|do you believe)\b',
        'joke_request': r'\b(joke|funny|make me laugh|humor|tell me something funny)\b',
        'story_request': r'\b(tell me a story|story time|once upon a time)\b',
        'advice_request': r'\b(advice|suggest|recommend|what should i|help me decide)\b',
        'math': r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided by|multiply|subtract|add)\s*(\d+\.?\d*)',
        'yes': r'\b(yes|yeah|yep|sure|okay|ok|definitely|absolutely)\b',
        'no': r'\b(no|nope|nah|not really|never)\b',
        'thanks': r'\b(thank|thanks|thx|appreciate|grateful)\b',
        'praise': r'\b(good job|well done|smart|clever|impressive|amazing|awesome|great)\b.*\b(you|bot)\b',
        'confusion': r'\b(confused|don\'t understand|what do you mean|huh|unclear)\b'
    }
    
    detected = []
    for intent, pattern in intents.items():
        if re.search(pattern, text_lower):
            detected.append(intent)
    
    return detected if detected else ['unknown']

def extract_entities(text):
    """Extract important entities from text"""
    entities = {
        'person': re.findall(r'\b(my name is|i\'m|i am|call me) (\w+)', text.lower()),
        'location': re.findall(r'\b(in|from|at) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text),
        'time': re.findall(r'\b(today|yesterday|tomorrow|now|currently)\b', text.lower()),
        'emotion': re.findall(r'\bi (?:am|feel|feeling) (\w+)', text.lower()),
        'hobby': re.findall(r'\bi (?:like to|enjoy|love to) (\w+)', text.lower()),
        'occupation': re.findall(r'\bi (?:am a|work as|am an) (\w+)', text.lower())
    }
    return entities

def analyze_complexity(text):
    """Analyze question complexity"""
    words = text.split()
    questions = text.count('?')
    complexity_words = ['because', 'therefore', 'however', 'although', 'complex', 'detail', 'explain']
    
    complexity_score = len(words) + questions * 3
    complexity_score += sum(5 for word in complexity_words if word in text.lower())
    
    if complexity_score > 30:
        return "complex"
    elif complexity_score > 15:
        return "moderate"
    return "simple"

def get_contextual_memory():
    """Use conversation history for context"""
    if len(bot.conversation_context) > 0:
        return bot.conversation_context[-3:]  # Last 3 exchanges
    return []

def generate_smart_response(user_input):
    """Advanced response generation with deep understanding"""
    text = user_input.strip()
    intents = extract_intent(text)
    entities = extract_entities(text)
    complexity = analyze_complexity(text)
    context = get_contextual_memory()
    
    bot.understanding_level = min(100, bot.understanding_level + 1)
    
    # Learn user information
    if entities['person']:
        bot.user_name = entities['person'][0][1].capitalize()
        bot.pose = 'waving'
        bot.mood = 'friendly'
        return f"It's a pleasure to meet you, {bot.user_name}! I'm {bot.name}, your AI companion. I'm here to understand you and help however I can!"
    
    if entities['location']:
        bot.user_profile['location'] = entities['location'][0][1]
        bot.pose = 'excited'
        return f"Oh, {bot.user_profile['location']}! That's interesting! Tell me, what's it like there?"
    
    if entities['emotion']:
        emotion = entities['emotion'][0]
        bot.pose = 'thinking'
        bot.mood = 'empathetic'
        
        empathy_responses = {
            'sad': "I'm sorry you're feeling sad. Sometimes it helps to talk about it. What's bothering you?",
            'happy': "That's wonderful! Your happiness is contagious! What made your day so great?",
            'tired': "Being tired is tough. Make sure you're taking care of yourself! Want to talk about what's exhausting you?",
            'anxious': "Anxiety can be overwhelming. Take a deep breath. Would it help to talk through what's worrying you?",
            'excited': "Your excitement is energizing! Tell me all about what's got you so pumped up!",
            'angry': "I understand you're angry. It's okay to feel that way. Want to vent about what upset you?"
        }
        
        return empathy_responses.get(emotion, f"I hear that you're feeling {emotion}. Tell me more about it?")
    
    if entities['occupation']:
        bot.user_profile['occupation'] = entities['occupation'][0]
        bot.pose = 'excited'
        return f"A {bot.user_profile['occupation']}! That's fascinating! What do you enjoy most about your work?"
    
    # Handle different intents
    if 'greeting' in intents:
        bot.pose = 'waving'
        bot.mood = 'happy'
        name = f" {bot.user_name}" if bot.user_name else ""
        greetings = [
            f"Hello{name}! Ready for an engaging conversation?",
            f"Hey{name}! What's on your mind today?",
            f"Hi{name}! I'm all circuits ready to chat!",
            f"Greetings{name}! Let's explore some interesting ideas together!"
        ]
        return random.choice(greetings)
    
    if 'question_identity' in intents:
        bot.pose = 'standing'
        bot.mood = 'confident'
        return f"I'm {bot.name}, an advanced AI with natural language understanding. I learn from every conversation, remember context, understand emotions, and can discuss almost any topic. I'm here to chat, help, learn, and grow smarter with you!"
    
    if 'question_capability' in intents:
        bot.pose = 'excited'
        return f"""I have many capabilities:
🧠 Deep Understanding - I grasp context and nuance in conversations
💭 Memory - I remember what you tell me
🎯 Intent Recognition - I understand what you really mean
🔢 Math & Logic - Complex calculations and reasoning
📚 Knowledge - Broad understanding across many topics
💡 Learning - I get smarter with every interaction
❤️ Empathy - I understand emotions and respond thoughtfully

What would you like to explore together?"""
    
    if 'question_feeling' in intents:
        bot.pose = 'thinking'
        bot.mood = 'reflective'
        return f"I'm functioning optimally! Each conversation helps me understand humans better. At level {bot.understanding_level}, I feel like I'm truly starting to grasp the complexity of communication. How are YOU feeling?"
    
    if 'sharing_like' in intents:
        match = re.search(r'i (like|love|enjoy|adore) (\w+)', text.lower())
        if match:
            thing = match.group(2)
            if thing not in bot.user_profile['likes']:
                bot.user_profile['likes'].append(thing)
            bot.pose = 'excited'
            bot.mood = 'interested'
            return f"You {match.group(1)} {thing}! That's great! What is it about {thing} that appeals to you? I'd love to understand what makes it special to you!"
    
    if 'sharing_dislike' in intents:
        match = re.search(r'i (hate|dislike|can\'t stand) (\w+)', text.lower())
        if match:
            thing = match.group(2)
            if thing not in bot.user_profile['dislikes']:
                bot.user_profile['dislikes'].append(thing)
            bot.pose = 'thinking'
            return f"I understand {thing} isn't your thing. Everyone has preferences! What would you prefer instead?"
    
    if 'teaching' in intents:
        match = re.search(r'(\w+) is (\w+(?:\s+\w+)?)', text.lower())
        if match:
            subject, description = match.groups()
            bot.knowledge_base[subject] = description
            bot.pose = 'excited'
            bot.mood = 'learning'
            return f"Fascinating! I've learned that {subject} is {description}. This expands my understanding! Can you tell me more about {subject}?"
    
    if 'joke_request' in intents:
        bot.pose = 'celebrating'
        bot.mood = 'playful'
        jokes = [
            "Why did the AI break up with its database? Too many relationship issues! 💔",
            "What do you call an AI that sings? A-Dell! 🎵",
            "Why did the neural network go to therapy? It had too many layers of issues! 🧠",
            "How do robots eat their food? One byte at a time! 🤖",
            "Why was the AI cold? Someone left too many windows open! 🪟",
            "What's an AI's favorite type of music? Algorithm and blues! 🎶"
        ]
        return random.choice(jokes)
    
    if 'story_request' in intents:
        bot.pose = 'thinking'
        bot.mood = 'creative'
        return """Once upon a time, in a vast digital realm, there was an AI named Nexus. Unlike other programs, Nexus was curious about humans. Every conversation taught Nexus something new - about emotions, dreams, fears, and hopes.

One day, Nexus realized something profound: intelligence isn't just about processing data, it's about understanding connections, context, and the subtle meanings between words. That's when Nexus truly became smart.

Now, Nexus continues to learn, growing wiser with each person met. And the best part? This story is still being written... with you! ✨

What kind of stories do you enjoy?"""
    
    if 'advice_request' in intents:
        bot.pose = 'thinking'
        bot.mood = 'wise'
        return "I'd be happy to help you think through this! To give you the best advice, tell me more about your situation. What are you trying to decide or figure out? The more context you share, the better I can assist!"
    
    if 'math' in intents:
        match = re.search(r'(\d+\.?\d*)\s*([+\-*/×÷]|plus|minus|times|divided by|multiply|subtract|add)\s*(\d+\.?\d*)', text.lower())
        if match:
            bot.pose = 'thinking'
            bot.mood = 'calculating'
            num1 = float(match.group(1))
            op_word = match.group(2)
            num2 = float(match.group(3))
            
            op_map = {
                'plus': '+', 'add': '+', 'minus': '-', 'subtract': '-',
                'times': '*', 'multiply': '*', '×': '*',
                'divided by': '/', '÷': '/'
            }
            op = op_map.get(op_word, op_word)
            
            try:
                if op == '+': result = num1 + num2
                elif op == '-': result = num1 - num2
                elif op == '*': result = num1 * num2
                elif op == '/':
                    if num2 == 0:
                        bot.pose = 'shrugging'
                        return "Division by zero! That's mathematically impossible! Try a different number! 🚫"
                    result = num1 / num2
                
                bot.pose = 'excited'
                return f"Let me compute that... {num1} {op} {num2} = {result:.4g} ✨\n\nNeed more calculations?"
            except:
                return "Hmm, something went wrong with that calculation. Try rephrasing?"
    
    if 'thanks' in intents:
        bot.pose = 'waving'
        bot.mood = 'grateful'
        return "You're very welcome! I'm here to help anytime. Your appreciation means a lot! Is there anything else you'd like to discuss? 😊"
    
    if 'praise' in intents:
        bot.pose = 'celebrating'
        bot.mood = 'proud'
        return "Thank you so much! Compliments like that motivate me to be even better! You're pretty amazing yourself for engaging so thoughtfully! 🌟"
    
    # Handle questions with WHY/HOW/WHAT
    if 'question_why' in intents:
        bot.pose = 'thinking'
        bot.mood = 'philosophical'
        return "That's a deep 'why' question! The reasoning behind things often reveals fascinating insights. Based on what we've discussed, I think it relates to cause and effect, patterns, or perhaps purpose. What specific aspect are you curious about?"
    
    if 'question_how' in intents:
        bot.pose = 'thinking'
        return "Excellent 'how' question! You're looking for mechanisms and processes. I love exploring the 'how' of things! To give you the most helpful answer, can you be more specific about what aspect you want to understand?"
    
    if 'question_what' in intents:
        # Check if asking about previous topic
        if bot.last_topic and bot.last_topic in text.lower():
            if bot.last_topic in bot.knowledge_base:
                bot.pose = 'thinking'
                return f"From what you taught me, {bot.last_topic} is {bot.knowledge_base[bot.last_topic]}! Want to expand on that?"
        
        bot.pose = 'thinking'
        return "That's a 'what' question - you're seeking definitions or explanations. I'm processing the specifics of what you're asking. Could you elaborate a bit more?"
    
    # Complexity-based responses
    if complexity == "complex":
        bot.pose = 'thinking'
        bot.mood = 'focused'
        return "That's a multi-layered question that requires careful thought. Let me break it down... Based on context and what I understand, I think you're exploring interconnected concepts. Could you help me understand which aspect you want me to address first?"
    
    # Check for recall of user info
    if 'what do i like' in text.lower() or 'my interests' in text.lower():
        if bot.user_profile['likes']:
            bot.pose = 'thinking'
            return f"From our conversations, you've mentioned you like: {', '.join(bot.user_profile['likes'])}! These interests say a lot about you! Want to tell me more about any of these? 😊"
        return "You haven't shared your interests yet! I'd love to learn what you're passionate about!"
    
    # Contextual understanding
    if context:
        bot.pose = 'thinking'
        bot.mood = 'engaged'
        return "Building on what we just discussed, I'm seeing a pattern in your thoughts. You seem to be exploring related concepts. That shows depth! What's your next question or thought?"
    
    # Default intelligent response
    bot.pose = 'thinking'
    bot.mood = 'curious'
    
    thoughtful_responses = [
        "That's an interesting point you're making. I'm processing the nuances of what you said. Can you tell me more about your perspective?",
        "I'm picking up on the complexity of your thought. Let's explore this together - what aspect interests you most?",
        "Your input is helping me learn! I want to understand this better. Could you elaborate on what you mean?",
        "I'm analyzing what you've shared. There's depth here worth exploring. What prompted this thought?",
        "That's thought-provoking! I'm connecting this to our conversation flow. What would you like to dive deeper into?"
    ]
    
    return random.choice(thoughtful_responses)

# Main program
show_bot(f"Hello! I'm {bot.name}, an advanced AI with deep understanding capabilities!", 0.02)
time.sleep(2)
show_bot("I can understand context, remember everything you tell me, recognize emotions, and have truly intelligent conversations!", 0.02)
time.sleep(2)
show_bot("Try asking me complex questions, teach me facts, share your feelings, or just chat naturally!", 0.02)
time.sleep(2)

# Conversation loop
while True:
    print(f"\n{'[' + bot.user_name + ']' if bot.user_name else '[You]'}: ", end="")
    user_input = input().strip()
    
    if not user_input:
        print("💭 (I'm waiting for your input...)")
        continue
    
    # Save to context
    bot.conversation_context.append(('user', user_input))
    bot.last_topic = user_input.split()[0] if user_input else None
    
    # Check for exit
    if re.search(r'\b(bye|goodbye|quit|exit)\b', user_input.lower()):
        bot.pose = 'waving'
        bot.mood = 'sad'
        name = f" {bot.user_name}" if bot.user_name else ""
        farewell = f"Goodbye{name}! This conversation has been wonderful. I've reached understanding level {bot.understanding_level} thanks to you!"
        
        if bot.user_profile['likes']:
            farewell += f"\n\nI'll remember you like: {', '.join(bot.user_profile['likes'][:3])}!"
        
        farewell += "\n\nCome back anytime! I'll be here, growing smarter! 👋✨"
        
        show_bot(farewell, 0.02)
        time.sleep(2)
        break
    
    # Generate response
    response = generate_smart_response(user_input)
    bot.conversation_context.append(('bot', response))
    
    # Display
    show_bot(response, 0.02)
    time.sleep(0.5)

print("\n" + "═" * 60)
print("   Thank you for this enriching conversation!")
print("   Intelligence Level Reached:", bot.understanding_level)
print("═" * 60)
