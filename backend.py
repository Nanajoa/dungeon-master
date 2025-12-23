from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
print("Loading model...")
CUSTOM_MODEL_PATH = "models/reddit_adventure_gpt_final"
BASE_MODEL_PATH = "gpt2"

# Global variables for model management
current_model = None
current_tokenizer = None
current_model_name = None
base_model = None
base_tokenizer = None
poor_response_count = 0
USE_BASE_MODEL_THRESHOLD = 2  # Switch to base after 2 poor responses (lowered from 3)
FORCE_BASE_MODEL = False  # Set to True to always use base GPT-2

def load_custom_model():
    """Load the custom trained model"""
    global current_model, current_tokenizer, current_model_name
    
    # Check if forced to use base model
    if FORCE_BASE_MODEL:
        print("⚙️ FORCE_BASE_MODEL is enabled - skipping custom model")
        return False
    
    try:
        print("Attempting to load custom trained model...")
        tokenizer = GPT2Tokenizer.from_pretrained(CUSTOM_MODEL_PATH)
        
        # Try to load with PyTorch first, then safetensors
        try:
            model = GPT2LMHeadModel.from_pretrained(CUSTOM_MODEL_PATH, use_safetensors=False)
            print("✓ Custom model loaded from PyTorch format")
        except:
            model = GPT2LMHeadModel.from_pretrained(CUSTOM_MODEL_PATH, use_safetensors=True)
            print("✓ Custom model loaded from SafeTensors format")
        
        current_model = model
        current_tokenizer = tokenizer
        current_model_name = "reddit_adventure_gpt_final"
        return True
    except Exception as e:
        print(f"❌ Custom model loading failed: {e}")
        return False

def load_custom_model():
    """Load the custom trained model"""
    global current_model, current_tokenizer, current_model_name
    try:
        print("Attempting to load custom trained model...")
        tokenizer = GPT2Tokenizer.from_pretrained(CUSTOM_MODEL_PATH)
        
        # Try to load with PyTorch first, then safetensors
        try:
            model = GPT2LMHeadModel.from_pretrained(CUSTOM_MODEL_PATH, use_safetensors=False)
            print("✓ Custom model loaded from PyTorch format")
        except:
            model = GPT2LMHeadModel.from_pretrained(CUSTOM_MODEL_PATH, use_safetensors=True)
            print("✓ Custom model loaded from SafeTensors format")
        
        current_model = model
        current_tokenizer = tokenizer
        current_model_name = "reddit_adventure_gpt_final"
        return True
    except Exception as e:
        print(f"❌ Custom model loading failed: {e}")
        return False

def load_base_model():
    """Load the base GPT-2 model"""
    global current_model, current_tokenizer, current_model_name
    try:
        print("Loading base GPT-2 model...")
        tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        
        current_model = model
        current_tokenizer = tokenizer
        current_model_name = "gpt2-base"
        print("✓ Base GPT-2 model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Base model loading failed: {e}")
        return False

# Try to load custom model first, fall back to base if needed
if not load_custom_model():
    print("⚠️ Custom model unavailable, using base GPT-2")
    load_base_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
current_model.to(device)
current_model.eval()

# Also prepare base model as backup if custom model is loaded
if current_model_name == "reddit_adventure_gpt_final":
    print("Preparing base GPT-2 as backup...")
    try:
        base_tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
        base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.to(device)
        base_model.eval()
        print("✓ Base model ready as backup")
    except:
        print("⚠️ Could not load backup model")

print(f"\n{'='*60}")
print(f"✓ Primary Model: {current_model_name}")
print(f"✓ Device: {device}")
print(f"✓ Backup Model: {'Available' if base_model else 'None'}")
print(f"{'='*60}\n")

def extract_game_state_changes(text):
    """Extract potential game state changes from generated text"""
    changes = {
        'health_change': 0,
        'new_items': [],
        'location_hints': []
    }
    
    text_lower = text.lower()
    
    # Detect damage/healing
    damage_keywords = ['hurt', 'wounded', 'injured', 'damage', 'hit', 'attacked', 'pain']
    heal_keywords = ['heal', 'rest', 'recover', 'potion', 'bandage', 'better']
    
    if any(word in text_lower for word in damage_keywords):
        changes['health_change'] = -15
    elif any(word in text_lower for word in heal_keywords):
        changes['health_change'] = 20
    
    # Detect items
    item_patterns = [
        r'find.*?(\w+)',
        r'discover.*?(\w+)',
        r'pick.*?up.*?(\w+)',
        r'take.*?(\w+)',
        r'obtain.*?(\w+)'
    ]
    
    for pattern in item_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) > 3 and match not in ['the', 'and', 'you', 'your']:
                changes['new_items'].append(match)
    
    # Detect locations
    location_keywords = ['enter', 'arrive', 'reach', 'walk to', 'go to', 'move to']
    if any(word in text_lower for word in location_keywords):
        # Extract potential location names (capitalized words after location keywords)
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ['enter', 'arrive', 'reach'] and i < len(words) - 1:
                potential_loc = ' '.join(words[i+1:i+3])
                changes['location_hints'].append(potential_loc)
    
    return changes

def switch_to_base_model():
    """Switch to using the base GPT-2 model"""
    global current_model, current_tokenizer, current_model_name, poor_response_count
    
    if base_model is None:
        print("⚠️ Base model not available for switch")
        return False
    
    print("\n" + "!"*60)
    print("⚠️ SWITCHING TO BASE GPT-2 MODEL")
    print(f"   Reason: Custom model produced {poor_response_count} poor responses")
    print("!"*60 + "\n")
    
    current_model = base_model
    current_tokenizer = base_tokenizer
    current_model_name = "gpt2-base (switched)"
    poor_response_count = 0  # Reset counter
    return True

def generate_response(context, player_action, max_length=150):
    """Generate story response using the model"""
    global poor_response_count
    
    # Prepare prompts optimized for different models
    if "gpt2-base" in current_model_name:
        # Simpler prompts work better for base GPT-2
        prompts = [
            f"Story: {context}\n\nPlayer action: {player_action}\n\nWhat happens next:",
            f"{context} {player_action}",
            f"In this adventure: {context}\nThe hero decides to: {player_action}\nThe result:",
        ]
    else:
        # Custom model might understand special tokens
        prompts = [
            f"<|context|> {context} <|choice|> {player_action} <|response|>",
            f"Story: {context}\n\nPlayer: {player_action}\n\nDungeon Master:",
            f"Context: {context}\nAction: {player_action}\nResponse:",
        ]
    
    best_response = None
    best_score = 0
    
    for prompt in prompts:
        try:
            inputs = current_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
            
            with torch.no_grad():
                outputs = current_model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.85,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=current_tokenizer.eos_token_id,
                    eos_token_id=current_tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            generated = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response part
            if "<|response|>" in generated:
                response = generated.split("<|response|>")[-1].strip()
            elif "Dungeon Master:" in generated:
                response = generated.split("Dungeon Master:")[-1].strip()
            elif "What happens next:" in generated:
                response = generated.split("What happens next:")[-1].strip()
            elif "The result:" in generated:
                response = generated.split("The result:")[-1].strip()
            elif "Response:" in generated:
                response = generated.split("Response:")[-1].strip()
            else:
                response = generated[len(prompt):].strip()
            
            # Clean up
            response = response.split('<|')[0].strip()
            response = response.split('\n\n')[0].strip()  # Take first paragraph
            
            # Remove any remaining prompt fragments
            if "Context:" in response or "Action:" in response or "Story:" in response:
                continue
            
            # Score the response
            score = evaluate_response_quality(response, player_action)
            
            if score > best_score:
                best_score = score
                best_response = response
        
        except Exception as e:
            print(f"Error with prompt format: {e}")
            continue
    
    # Check if response quality is acceptable
    if not best_response or best_score < 3 or len(best_response) < 20:
        print(f"⚠️ Poor response - score: {best_score}, length: {len(best_response) if best_response else 0}, model: {current_model_name}")
        poor_response_count += 1
        
        # Try switching to base model if custom model is consistently bad
        if poor_response_count >= USE_BASE_MODEL_THRESHOLD and "gpt2-base" not in current_model_name and base_model:
            switch_to_base_model()
            # Retry with base model
            return generate_response(context, player_action, max_length)
        
        return generate_smart_fallback(context, player_action)
    
    # Good response - reset counter
    if poor_response_count > 0:
        poor_response_count = max(0, poor_response_count - 1)
    
    print(f"✓ Good response - score: {best_score}, model: {current_model_name}")
    return best_response

def evaluate_response_quality(response, action):
    """Score response quality (0-10) - STRICT evaluation"""
    if not response or len(response) < 10:
        return 0
    
    score = 5  # Base score
    
    # Check length (should be reasonable)
    if 50 < len(response) < 300:
        score += 2
    elif len(response) < 20:
        score -= 4
    elif len(response) > 400:  # Too long is also bad
        score -= 2
    
    # STRICT: Check for severe incoherence markers
    severe_incoherent_patterns = [
        'the the', 'and and', 'you you', 'we we',
        '...', '..."', '"..."', 
        'dragon dragon', 'speak speak',
        'aliens or something', 'what better way',
        'face it', "let's face",
        'pretty dumb', 'whole thing was',
        '[of]', '[', ']',  # Brackets indicate model confusion
        'i\'ll need your help', 'we\'ve got',  # Breaking 3rd person
        'he tells me', 'my friend', 'walks away from us',  # Wrong perspective
    ]
    
    response_lower = response.lower()
    for pattern in severe_incoherent_patterns:
        if pattern in response_lower:
            print(f"   ❌ Found incoherent pattern: '{pattern}'")
            score -= 5  # Heavy penalty
    
    # Check for first/second person confusion in narration
    problematic_phrases = ['he tells me', 'she tells me', 'we both', 'let me', 'i\'ll', 'my friend', 'our path']
    for phrase in problematic_phrases:
        if phrase in response_lower:
            print(f"   ❌ Perspective issue: '{phrase}'")
            score -= 4
    
    # Check if response contains ellipsis spam
    if response.count('...') > 1 or response.count('…') > 1:
        print(f"   ❌ Too many ellipses")
        score -= 3
    
    # Check for weird punctuation patterns
    if '!  ' in response or '?  ' in response or ',  ' in response:
        print(f"   ❌ Weird spacing")
        score -= 2
    
    # Check if it's relevant to the action
    action_words = action.lower().split()
    relevance = sum(1 for word in action_words if word in response_lower and len(word) > 3)
    score += min(relevance, 2)
    
    # Penalize if it seems to be continuing the context instead of responding
    if 'context' in response_lower or '<|' in response:
        print(f"   ❌ Contains meta text")
        score -= 4
    
    # Check for complete sentences
    if response.endswith(('.', '!', '?', '"')):
        score += 1
    else:
        score -= 2
    
    # Check if response starts appropriately (should start with action/description)
    first_words = response.lower()[:30]
    if any(word in first_words for word in ["i'll", "we've", "let me", "my friend", "he tells"]):
        print(f"   ❌ Bad opening")
        score -= 4
    
    return max(0, min(10, score))

def generate_smart_fallback(context, action):
    """Generate contextually aware, engaging fallback responses with clear guidance"""
    action_lower = action.lower()
    context_lower = context.lower()
    
    # Location-based responses with CLEAR next steps
    if 'tavern' in context_lower:
        if 'look' in action_lower or 'examine' in action_lower:
            return "The tavern is nearly empty. A few patrons huddle in dark corners, avoiding eye contact. The bartender wipes down the counter nervously. You notice: a back door to the alley, stairs to the rooms above, and the bartender who might have information. You could talk to the bartender, search the room where you woke up, or head outside to track the thief."
        elif 'bartender' in action_lower or 'talk' in action_lower:
            return "The bartender leans in close. 'That hooded stranger? Left right after you passed out, heading toward the eastern gate. Word is they're connected to the Shadow Market - black market dealers who operate in the old warehouse district. You'll need to be careful; they don't take kindly to outsiders asking questions.'"
        elif 'go' in action_lower or 'leave' in action_lower or 'exit' in action_lower or 'outside' in action_lower:
            return "You step into the street. Dawn breaks over the city. To the east, you see the merchant quarter bustling with early morning activity. To the west, narrow alleyways lead to the warehouse district - rougher territory, but likely where the thief would fence stolen goods. Which way do you go?"
        elif 'search' in action_lower or 'room' in action_lower:
            return "You search the room where you woke up. Under the bed, you find a torn piece of dark fabric - likely from the thief's cloak. There's also a strange symbol drawn in chalk on the floor: a serpent eating its tail. This might be important. You should ask around about this symbol."
    
    elif 'station' in context_lower or 'space' in context_lower or 'alpha' in context_lower:
        if 'engineering' in action_lower or 'life support' in action_lower or 'go' in action_lower:
            return "You pull yourself up and head toward the engineering section. The corridor is dark, lit only by emergency strips. Warning klaxons echo distantly. You reach a sealed bulkhead door marked 'ENGINEERING - LIFE SUPPORT CONTROL'. The access panel is dead, but you notice a manual override lever and a maintenance shaft entrance nearby. You could try forcing the override or take the maintenance shaft."
        elif 'look' in action_lower or 'examine' in action_lower:
            return "The medical bay shows signs of a hasty evacuation. Equipment is knocked over, medical supplies scattered. Through the viewport, Earth looks peaceful - unaware of your crisis. You spot: a functioning computer terminal, a locker that might contain tools, and the door to the main corridor. You need to access life support controls in engineering, but you'll need to figure out how to get there."
        elif 'terminal' in action_lower or 'computer' in action_lower:
            return "You access the terminal. ERROR messages flash across the screen. Life support: 35% and dropping. Last log entry: 'Hull breach in Sector 7. Emergency evacuation initiated. AI systems behaving erratically - DO NOT TRUST CENTRAL AI.' A schematic shows engineering is three decks down. You'll need to move fast."
        elif 'ai' in action_lower or 'speak' in action_lower:
            return "You try to communicate with the station AI. Its voice crackles with static: 'User... not recognized. Life support... failing. Recommend... immediate evacuation... but... but... WHERE WOULD YOU GO?' Something is clearly wrong with the AI. You should avoid relying on it and head to engineering manually."
        elif 'next' in action_lower or 'continue' in action_lower:
            return "Time is running out. The air is getting thinner. You must decide: head directly to engineering through the main corridors (faster but potentially dangerous), or take the safer maintenance shafts (slower but might avoid whatever caused the crew to evacuate). What do you do?"
    
    elif 'cabin' in context_lower or 'forest' in context_lower or 'wood' in context_lower:
        if 'look' in action_lower or 'examine' in action_lower:
            return "The cabin is in shambles. Blood trails lead into the forest. Strange symbols are carved into the trees - they look ancient, deliberate. Your camping supplies are scattered. Your phone is completely dead. You find a hunting knife, some rope, and a flashlight with weak batteries. Through the trees, you hear something large moving. You need to either follow the blood trail to find your friends, or find a defensible position."
        elif 'follow' in action_lower or 'trail' in action_lower or 'forest' in action_lower:
            return "You follow the blood trail deeper into the forest, knife in hand. The trees grow denser, blocking out the morning light. The trail leads to a clearing where you find torn clothing and more blood, but no bodies. The tracks continue in two directions: north toward an old mine entrance, or east toward what sounds like running water. You hear a branch snap behind you."
        elif 'hide' in action_lower or 'wait' in action_lower or 'stay' in action_lower:
            return "You crouch behind an overturned table, knife ready. Heavy footsteps circle the cabin. Through a gap in the boards, you glimpse something massive and humanoid, but wrong - too tall, limbs too long. It sniffs the air, then moves away toward the forest. You've got a brief window to make your move. Do you stay hidden or make a run for it?"
    
    elif 'office' in context_lower or 'midnight' in context_lower or 'caller' in context_lower:
        if 'look' in action_lower or 'examine' in action_lower:
            return "Your office is cluttered with case files. Rain hammers the window. On your desk: the case file for the Blackwell murder (the secret you buried), your phone, and a envelope that wasn't here before. Inside the envelope is a photograph of you at the crime scene that night. Someone knows. You should examine your files or trace the call."
        elif 'call' in action_lower or 'trace' in action_lower or 'phone' in action_lower:
            return "You check your phone records. The call came from a burner phone, last pinged near the old Riverside Warehouse - the same place the Blackwell murder happened five years ago. Someone is sending you a message. You could go there now to confront them, or search your files first to remember exactly what happened that night."
    
    # Generic action-based responses with clear options
    if 'search' in action_lower or 'find' in action_lower:
        return "You search carefully and find something useful: a clue that points you in the right direction. You have a clearer picture now. You can continue investigating, move to a new location, or talk to someone who might know more."
    elif 'attack' in action_lower or 'fight' in action_lower:
        return "You engage in combat! The fight is intense. You manage to defend yourself but take some damage in the process. Your opponent retreats, wounded. You can pursue them, search the area, or tend to your wounds and regroup."
    elif 'run' in action_lower or 'flee' in action_lower:
        return "You turn and run! Adrenaline surges as you sprint to safety. You lose your pursuer by ducking through a narrow passage. You're safe for now, but you've lost ground. You need to decide on your next move quickly."
    elif 'next' in action_lower or 'continue' in action_lower:
        return "You press forward, determined to see this through. The path ahead is challenging but you're making progress. New obstacles emerge - you'll need to stay sharp and think carefully about each decision."
    
    # Default with clear guidance
    return "Your action changes the situation. You assess your surroundings and options. You can: explore the area more thoroughly, interact with someone or something nearby, or move to a different location. What's your next move?"

# ============= ROUTES =============

@app.route('/')
def serve_frontend():
    """Serve the main HTML file"""
    try:
        return send_file('index.html')
    except FileNotFoundError:
        return jsonify({
            'error': 'Frontend not found',
            'message': 'Please ensure index.html is in the same directory as backend.py'
        }), 404

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': current_model_name,
        'device': device,
        'backup_available': base_model is not None,
        'poor_response_count': poor_response_count
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate story responses"""
    try:
        data = request.json
        context = data.get('context', '')
        player_action = data.get('player_action', '')
        max_length = data.get('max_length', 150)
        
        if not player_action:
            return jsonify({'error': 'No player action provided'}), 400
        
        # Generate response
        response = generate_response(context, player_action, max_length)
        
        # Extract game state changes
        state_changes = extract_game_state_changes(response)
        
        return jsonify({
            'response': response,
            'state_changes': state_changes
        })
    
    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎮 AI DUNGEON MASTER BACKEND STARTING")
    print("="*60)
    print(f"✓ Primary Model: {current_model_name}")
    print(f"✓ Backup Model: {'Available' if base_model else 'None'}")
    print(f"✓ Device: {device}")
    print(f"✓ Server: http://localhost:5000")
    print(f"✓ Auto-switch: Will switch to base GPT-2 after {USE_BASE_MODEL_THRESHOLD} poor responses")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)