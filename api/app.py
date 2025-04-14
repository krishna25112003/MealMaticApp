from flask import Flask, request, jsonify, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from serverless_wsgi import handle_request
app = Flask(__name__)

# Initialize RAG components
print("🚀 Initializing RAG system...")

# 1. Load Dataset
dataset_path = "Mealmatic Dataset.xlsx"
df = pd.read_excel(dataset_path)
df.columns = df.columns.str.strip()

# 2. Prepare Documents
documents = []
for idx, row in df.iterrows():
    doc = {
        "meal_category": row["Meal Category"],
        "food_name": row["food_name"],
        "serving_size": row["servings_unit"],
        "calories": row["unit_serving_energy_kcal"],
        "protein": row["unit_serving_protein_g"],
        "carbs": row["unit_serving_carb_g"],
        "fats": row["unit_serving_fat_g"],
        "fibre": row["unit_serving_fibre_g"],
        "freesugar": row["unit_serving_freesugar_g"]
    }
    documents.append(doc)

# 3. Create Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
document_texts = [json.dumps(doc) for doc in documents]
document_embeddings = embedder.encode(document_texts, convert_to_tensor=False)

# 4. Build FAISS Index
dimension = document_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

print("✅ RAG system initialized!")

# Mistral API Configuration
MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_TOKEN = "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"  # Use your actual token

def retrieve_similar_docs(query, k=5):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in I[0]]

def calculate_tdee(user_profile):
    try:
        weight = float(user_profile['weight'])
        height = float(user_profile['height'])
        age = int(user_profile['age'])
        gender = user_profile['gender'].lower()
        
        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
            
        activity_multipliers = {
            "1": 1.2, "2": 1.375, "3": 1.55, "4": 1.725, "5": 1.9
        }
        
        multiplier = activity_multipliers.get(user_profile['activity_level'], 1.55)
        tdee = bmr * multiplier
        
        if user_profile['fitness_goal'] == "1":  # Weight loss
            target_calories = tdee - 500
        elif user_profile['fitness_goal'] == "2":  # Muscle gain
            target_calories = tdee + 300
        else:
            target_calories = tdee
            
        return int(target_calories)
    except:
        return 2000  # Default value

def construct_llm_prompt(user_profile, context_docs):
    # Map numeric inputs to text
    activity_map = {
        "1": "Sedentary", "2": "Lightly active", "3": "Moderately active",
        "4": "Very active", "5": "Super active"
    }
    diet_map = {
        "1": "Vegetarian", "2": "Non-Vegetarian", "3": "Vegan", "4": "No restrictions"
    }
    goal_map = {
        "1": "Weight loss", "2": "Muscle gain", 
        "3": "Maintenance", "4": "Improve overall health"
    }
    health_map = {
        "1": "Diabetes", "2": "Hypertension", 
        "3": "Heart disease", "4": "High cholesterol", "5": "None"
    }
    
    # Format context
    context = "\n".join([f"{doc['food_name']} ({doc['meal_category']}): {doc['calories']} kcal" 
                         for doc in context_docs[:10]])  # Use top 10 docs
    
    prompt = f"""Create a detailed 1-day meal plan with these specifications:

User Profile:
- Age: {user_profile['age']}
- Gender: {user_profile['gender']}
- Height: {user_profile['height']} cm
- Weight: {user_profile['weight']} kg
- Activity Level: {activity_map.get(user_profile['activity_level'], "Moderately active")}
- Dietary Preferences: {', '.join([diet_map.get(d, d) for d in user_profile.get('dietary_prefs', [])])}
- Fitness Goal: {goal_map.get(user_profile['fitness_goal'], "Improve health")}
- Health Considerations: {', '.join([health_map.get(h, h) for h in user_profile.get('health_issues', [])])}
- Allergies: {user_profile.get('allergies', 'None')}
- Preferred Cuisines: {user_profile.get('cuisine_preference', 'Any')}

Target Daily Calories: {calculate_tdee(user_profile)} kcal

Available Foods:
{context}

Instructions:
1. Create a balanced meal plan with breakfast, morning snack, lunch, evening snack, and dinner
2. Include specific portion sizes
3. Provide nutritional info (calories, protein, carbs, fats) for each meal
4. Total daily calories should be near the target
5. Address any dietary restrictions or health concerns
6. Make suggestions practical and easy to prepare

the format should be like this

=== DAILY MEAL PLAN ===
Name: Sarah Miller
Goal: Weight Loss (1500 kcal/day)
Diet: Vegetarian
Allergies: Peanuts

----- BREAKFAST (320 kcal) -----
• 1/2 cup oatmeal cooked in almond milk
• 1 tbsp chia seeds
• 1/2 banana, sliced
• 1 tsp honey
• 1 hard-boiled egg

----- MORNING SNACK (180 kcal) -----
• 1 small apple
• 10 raw almonds
• 1 cup green tea

----- LUNCH (400 kcal) -----
• 1 cup quinoa salad with:
  - Cherry tomatoes
  - Cucumber
  - Feta cheese
  - Lemon-olive oil dressing
• 1 cup vegetable soup

----- AFTERNOON SNACK (150 kcal) -----
• 3/4 cup Greek yogurt
• 1/4 cup mixed berries
• 1 tsp flaxseeds

----- DINNER (450 kcal) -----
• 1 cup lentil curry
• 1/2 cup brown rice
• 1 cup steamed broccoli
• 1 tsp ghee

DAILY TOTALS:
Calories: 1500
Protein: 68g
Carbs: 185g
Fats: 52g

NOTES:
- Drink at least 8 glasses of water
- Optional: Add 1 tsp pumpkin seeds to yogurt
- Can substitute quinoa with bulgur wheat
- For faster weight loss, reduce rice portion by 1/4 cup

Format your response with clear headings for each meal and the nutritional information."""
    
    return prompt

def generate_with_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except requests.exceptions.RequestException as e:
        print(f"Error calling Mistral API: {e}")
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        try:
            # Get form data
            user_profile = {
                'name': request.form.get('name'),
                'age': request.form.get('age'),
                'gender': request.form.get('gender'),
                'height': request.form.get('height'),
                'weight': request.form.get('weight'),
                'activity_level': request.form.get('activity_level'),
                'dietary_prefs': request.form.getlist('dietary_prefs'),
                'fitness_goal': request.form.get('fitness_goal'),
                'health_issues': request.form.getlist('health_issues'),
                'allergies': request.form.get('allergies'),
                'cuisine_preference': request.form.get('cuisine_preference')
            }
            
            # RAG retrieval
            query = f"{user_profile['dietary_prefs']} {user_profile['fitness_goal']} meals"
            if user_profile['cuisine_preference']:
                query += f" with {user_profile['cuisine_preference']} cuisine"
            retrieved_docs = retrieve_similar_docs(query, k=10)
            
            # Construct prompt with retrieved docs
            prompt = construct_llm_prompt(user_profile, retrieved_docs)
            print(f"\n=== Generated Prompt ===\n{prompt}\n")
            
            # Call Mistral API
            meal_plan_text = generate_with_mistral(prompt)
            
            if not meal_plan_text:
                raise Exception("Failed to generate meal plan from Mistral API")
            
            # Format response
            response = {
                'status': 'success',
                'user_profile': user_profile,
                'meal_plan': {
                    'text': meal_plan_text,
                    'retrieved_docs': retrieved_docs[:3]  # For debugging
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    return render_template('generate.html')

# Add this to your existing app.py

@app.route('/workout')
def workout():
    workout_plans = [
        {
            "id": 1,
            "title": "Full Body Beginner",
            "duration": "30 mins",
            "difficulty": "Beginner",
            "focus": "Full Body",
            "description": "Perfect for starters - works all major muscle groups with basic exercises.",
            "exercises": [
                "Bodyweight Squats: 3x12",
                "Push-ups (knees if needed): 3x8",
                "Bent-over Rows (with bands): 3x10",
                "Plank: 3x20 sec",
                "Glute Bridges: 3x12"
            ]
        },
        {
            "id": 2,
            "title": "HIIT Fat Burner",
            "duration": "25 mins",
            "difficulty": "Intermediate",
            "focus": "Cardio",
            "description": "High intensity interval training for maximum calorie burn.",
            "exercises": [
                "Jump Squats: 40s work/20s rest x3",
                "Mountain Climbers: 40s work/20s rest x3",
                "Burpees: 40s work/20s rest x3",
                "High Knees: 40s work/20s rest x3",
                "Plank Jacks: 40s work/20s rest x3"
            ]
        },
        {
            "id": 3,
            "title": "Yoga Flow",
            "duration": "45 mins",
            "difficulty": "All Levels",
            "focus": "Flexibility",
            "description": "Gentle yoga sequence for flexibility and relaxation.",
            "exercises": [
                "Sun Salutations x5",
                "Warrior Series (I, II, III)",
                "Tree Pose (each side)",
                "Downward Dog Flow",
                "Savasana (5 mins)"
            ]
        },
        {
            "id": 4,
            "title": "Strength Training",
            "duration": "50 mins",
            "difficulty": "Advanced",
            "focus": "Muscle Building",
            "description": "Compound lifts for full-body strength development.",
            "exercises": [
                "Barbell Squats: 4x8",
                "Bench Press: 4x8",
                "Deadlifts: 3x6",
                "Overhead Press: 3x8",
                "Pull-ups: 3xMax"
            ]
        },
        {
            "id": 5,
            "title": "Core Crusher",
            "duration": "20 mins",
            "difficulty": "Intermediate",
            "focus": "Abs/Core",
            "description": "Targeted core workout for strong abs and back.",
            "exercises": [
                "Plank: 3x30s",
                "Russian Twists: 3x15/side",
                "Leg Raises: 3x12",
                "Bicycle Crunches: 3x20",
                "Superman Holds: 3x20s"
            ]
        },
        {
            "id": 6,
            "title": "Runner's Routine",
            "duration": "40 mins",
            "difficulty": "Intermediate",
            "focus": "Endurance",
            "description": "Improve running performance and endurance.",
            "exercises": [
                "Dynamic Warm-up: 10 mins",
                "Interval Runs: 30s sprint/90s jog x8",
                "Cool-down Stretches"
            ]
        },
        {
            "id": 7,
            "title": "Upper Body Blast",
            "duration": "35 mins",
            "difficulty": "Intermediate",
            "focus": "Arms/Back/Chest",
            "description": "Comprehensive upper body strength workout.",
            "exercises": [
                "Push-ups: 4x12",
                "Dumbbell Rows: 3x10/side",
                "Shoulder Press: 3x10",
                "Bicep Curls: 3x12",
                "Tricep Dips: 3x12"
            ]
        },
        {
            "id": 8,
            "title": "Lower Body Power",
            "duration": "30 mins",
            "difficulty": "Intermediate",
            "focus": "Legs/Glutes",
            "description": "Build strength and power in your lower body.",
            "exercises": [
                "Squats: 4x12",
                "Lunges: 3x10/leg",
                "Calf Raises: 3x15",
                "Romanian Deadlifts: 3x10",
                "Wall Sit: 3x30s"
            ]
        },
        {
            "id": 9,
            "title": "Post-Workout Stretch",
            "duration": "15 mins",
            "difficulty": "All Levels",
            "focus": "Recovery",
            "description": "Essential stretches to improve flexibility and recovery.",
            "exercises": [
                "Hamstring Stretch (each leg)",
                "Quad Stretch (each leg)",
                "Shoulder Cross-body Stretch",
                "Cat-Cow Stretch",
                "Child's Pose Hold"
            ]
        },
        {
            "id": 10,
            "title": "Pilates Fundamentals",
            "duration": "45 mins",
            "difficulty": "Beginner",
            "focus": "Core/Posture",
            "description": "Improve core strength and posture with Pilates basics.",
            "exercises": [
                "Hundred: 2x30s",
                "Roll Up: 3x8",
                "Single Leg Stretch: 3x10/side",
                "Swan Prep: 3x8",
                "Side Leg Series"
            ]
        },
        {
            "id": 11,
            "title": "Tabata Sprint",
            "duration": "20 mins",
            "difficulty": "Advanced",
            "focus": "HIIT",
            "description": "Short but intense full-body Tabata workout.",
            "exercises": [
                "Squat Jumps: 20s work/10s rest x8",
                "Push-ups: 20s work/10s rest x8",
                "Sit-ups: 20s work/10s rest x8",
                "Jump Lunges: 20s work/10s rest x8"
            ]
        },
        {
            "id": 12,
            "title": "Senior Fitness",
            "duration": "30 mins",
            "difficulty": "Beginner",
            "focus": "Mobility",
            "description": "Gentle exercises to maintain mobility and strength.",
            "exercises": [
                "Chair Squats: 3x10",
                "Wall Push-ups: 3x10",
                "Seated Leg Lifts: 3x8/leg",
                "Arm Circles: 3x10",
                "Balance Exercises"
            ]
        },
        {
            "id": 13,
            "title": "Pregnancy Safe",
            "duration": "25 mins",
            "difficulty": "Beginner",
            "focus": "Low Impact",
            "description": "Safe exercises for expecting mothers.",
            "exercises": [
                "Pelvic Tilts",
                "Modified Plank",
                "Wall Squats",
                "Seated Rows (with band)",
                "Kegel Exercises"
            ]
        },
        {
            "id": 14,
            "title": "Bodyweight Challenge",
            "duration": "30 mins",
            "difficulty": "Intermediate",
            "focus": "No Equipment",
            "description": "Effective workout using just your bodyweight.",
            "exercises": [
                "Push-up Variations: 3x10",
                "Squat Variations: 3x15",
                "Plank Variations: 3x30s",
                "Jumping Jacks: 3x30s",
                "Burpees: 3x8"
            ]
        },
        {
            "id": 15,
            "title": "Posture Correction",
            "duration": "20 mins",
            "difficulty": "Beginner",
            "focus": "Back/Shoulders",
            "description": "Exercises to correct rounded shoulders and improve posture.",
            "exercises": [
                "Band Pull-aparts: 3x12",
                "Face Pulls: 3x12",
                "Chin Tucks: 3x10",
                "Scapular Retractions: 3x10",
                "Thoracic Extensions"
            ]
        },
        {
            "id": 16,
            "title": "Kettlebell Routine",
            "duration": "40 mins",
            "difficulty": "Intermediate",
            "focus": "Full Body",
            "description": "Full-body workout using kettlebells.",
            "exercises": [
                "Kettlebell Swings: 3x15",
                "Goblet Squats: 3x12",
                "Turkish Get-ups: 3x5/side",
                "Kettlebell Rows: 3x10/side",
                "Kettlebell Press: 3x8/side"
            ]
        },
        {
            "id": 17,
            "title": "Morning Mobility",
            "duration": "15 mins",
            "difficulty": "All Levels",
            "focus": "Mobility",
            "description": "Wake up your body with these mobility exercises.",
            "exercises": [
                "Neck Rolls: 1 min",
                "Shoulder Rolls: 1 min",
                "Spinal Twists: 1 min/side",
                "Hip Circles: 1 min/side",
                "Ankle Rolls: 1 min/side"
            ]
        },
        {
            "id": 18,
            "title": "Dance Cardio",
            "duration": "45 mins",
            "difficulty": "Beginner",
            "focus": "Cardio",
            "description": "Fun dance-based cardio workout.",
            "exercises": [
                "Warm-up: 10 mins",
                "Dance Combinations: 30 mins",
                "Cool-down: 5 mins"
            ]
        },
        {
            "id": 19,
            "title": "Resistance Band",
            "duration": "30 mins",
            "difficulty": "Beginner",
            "focus": "Full Body",
            "description": "Effective workout using resistance bands.",
            "exercises": [
                "Band Squats: 3x12",
                "Band Rows: 3x12",
                "Band Chest Press: 3x12",
                "Band Lateral Raises: 3x12",
                "Band Glute Bridges: 3x12"
            ]
        },
        {
            "id": 20,
            "title": "Pre-Workout Warm-up",
            "duration": "10 mins",
            "difficulty": "All Levels",
            "focus": "Warm-up",
            "description": "Essential warm-up to prepare for any workout.",
            "exercises": [
                "Arm Circles: 1 min",
                "Leg Swings: 1 min/side",
                "Bodyweight Squats: 2x10",
                "Inchworms: 2x5",
                "Jumping Jacks: 1 min"
            ]
        }
    ]
    return render_template('workout.html', workout_plans=workout_plans)

if __name__ == '__main__':
    app.run(debug=True)

def handler(event, context):
    return handle_request(app, event, context)