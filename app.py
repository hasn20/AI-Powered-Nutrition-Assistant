import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Nutrition Assistant",
    page_icon="🥗",
    layout="wide"
)

# Define dietary preference options
DIETARY_PREFERENCES = [
    "No Restrictions", "Vegetarian", "Vegan", "Pescatarian", 
    "Gluten-Free", "Dairy-Free", "Keto", "Paleo", 
    "Mediterranean", "Low-Carb", "Low-Fat", "Other"
]

HEALTH_GOALS = [
    "Weight Loss", "Weight Gain", "Muscle Building", 
    "Maintenance", "Heart Health", "Diabetes Management", 
    "Energy Boost", "Better Sleep", "Digestive Health", "Other"
]

ALLERGIES = [
    "None", "Peanuts", "Tree Nuts", "Dairy", "Eggs", 
    "Fish", "Shellfish", "Soy", "Wheat", "Gluten"
]

ACTIVITY_LEVELS = [
    "Sedentary (little or no exercise)",
    "Lightly active (light exercise/sports 1-3 days/week)",
    "Moderately active (moderate exercise/sports 3-5 days/week)",
    "Very active (hard exercise/sports 6-7 days a week)",
    "Extra active (very hard exercise, physical job or training twice a day)"
]

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# Create a function to initialize the RAG pipeline
def initialize_rag_pipeline():
    if st.session_state.vectors is None:
        with st.spinner("Initializing nutrition knowledge base..."):
            try:
                # Use Hugging Face embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Load PDF documents from the knowledge_base directory
                loader = PyPDFDirectoryLoader("nutrition_knowledge_base")
                docs = loader.load()
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)
                
                # Create vector store
                vectors = FAISS.from_documents(final_documents, embeddings)
                
                st.session_state.vectors = vectors
                return True
            except Exception as e:
                st.error(f"Error initializing RAG pipeline: {str(e)}")
                return False
    return True

# Create a function to calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:  # Female
        return 10 * weight + 6.25 * height - 5 * age - 161

# Create a function to calculate daily calorie needs
def calculate_calorie_needs(bmr, activity_level):
    activity_multipliers = {
        "Sedentary (little or no exercise)": 1.2,
        "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
        "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
        "Very active (hard exercise/sports 6-7 days a week)": 1.725,
        "Extra active (very hard exercise, physical job or training twice a day)": 1.9
    }
    
    return bmr * activity_multipliers[activity_level]

# Create a function to adjust calories based on health goal
def adjust_calories_for_goal(calories, goal):
    if goal == "Weight Loss":
        return calories - 500  # Create a 500 calorie deficit
    elif goal == "Weight Gain" or goal == "Muscle Building":
        return calories + 500  # Add 500 calories for weight gain/muscle building
    else:
        return calories  # Maintenance or other health goals

# Create a function to get meal plan recommendations
def get_meal_plan_recommendations(age, gender, weight, height, activity_level, dietary_preference, health_goal, allergies, food_preferences, food_dislikes):
    if not initialize_rag_pipeline():
        return "Unable to generate meal plan due to an error in the RAG pipeline initialization."
    
    try:
        # Initialize the language model
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "GROQ API key not found. Please check your .env file."
        
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        
        # Calculate calorie needs
        bmr = calculate_bmr(weight, height, age, gender)
        daily_calories = calculate_calorie_needs(bmr, activity_level)
        adjusted_calories = adjust_calories_for_goal(daily_calories, health_goal)
        
        # Comprehensive prompt template for nutrition advice
        prompt = ChatPromptTemplate.from_template(
    """
    Personalized Evidence-Based Nutrition Plan

    **User Profile:**
    - Age: {age}
    - Gender: {gender}
    - Weight: {weight} kg
    - Height: {height} cm
    - Activity Level: {activity_level}
    - Dietary Preference: {dietary_preference}
    - Health Goal: {health_goal}
    - Allergies/Intolerances: {allergies}
    - Preferred Foods: {food_preferences}
    - Disliked Foods: {food_dislikes}
    - Estimated Daily Calorie Needs: {calories} calories

    **Comprehensive Nutrition Plan Instructions:**

    1. **Nutritional Analysis & Guidelines**
    - Provide macronutrient distribution appropriate for the user's health goal
    - Include micronutrient considerations based on their profile
    - Offer scientifically validated nutritional guidance for their specific health goal

    2. **Personalized 7-Day Meal Plan**
    - Create a complete 7-day meal plan with breakfast, lunch, dinner, and snacks
    - Ensure all meals align with dietary preferences and restrictions
    - Completely avoid any allergens listed
    - Include foods they prefer and exclude foods they dislike
    - Ensure caloric and nutritional targets are met
    - Include portion sizes and approximate calories per meal

    3. **Practical Implementation Guidance**
    - Provide meal prep strategies
    - Offer shopping list organized by food categories
    - Include time-saving preparation tips
    - Suggest meal timing recommendations based on activity level and goals

    4. **Educational Component**
    - Explain the nutritional science behind recommendations
    - Include information on key nutrients particularly important for their health goal
    - Provide tips for eating out while maintaining the plan

    5. **Progress Monitoring**
    - Suggest metrics to track progress
    - Recommend adjustment strategies as goals evolve
    - Provide guidelines for when to reassess the nutrition plan

    **Output Format:**
    - **Nutrition Strategy Summary:** [Brief overview of approach]
    - **Macronutrient Targets:** [Specific daily targets with rationale]
    - **7-Day Meal Plan:** [Detailed daily plans with all meals]
    - **Shopping List:** [Comprehensive list organized by category]
    - **Implementation Guide:** [Practical tips for success]
    - **Key Nutritional Insights:** [Educational content relevant to user]

    Utilize evidence-based nutrition science to provide a holistic, practical, and personalized nutrition plan.

    <context>
    {context}
    </context>
    """
)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate the query
        query = f"Create a personalized meal plan for a {age}-year-old {gender} with {dietary_preference} diet, aiming for {health_goal}, with allergies to {allergies}."
        
        # Get the response
        with st.spinner("Generating personalized meal plan..."):
            response = retrieval_chain.invoke({
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "dietary_preference": dietary_preference,
                "health_goal": health_goal,
                "allergies": allergies,
                "food_preferences": food_preferences,
                "food_dislikes": food_dislikes,
                "calories": round(adjusted_calories),
                "input": query
            })
            
        return response["answer"], response["context"]
        
    except Exception as e:
        return f"Error generating meal plan: {str(e)}", None

# Main application UI
st.title("🥗 AI-Powered Nutrition Assistant")
st.markdown("""
Provide your personal information and dietary preferences to get a
personalized, evidence-based meal plan that meets your nutritional needs and health goals!
""")

# Create two columns for the UI layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Personal Information")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.radio("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)
    height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
    activity_level = st.selectbox("Activity Level", ACTIVITY_LEVELS)
    
    st.header("Health Goals & Dietary Preferences")
    
    health_goal = st.selectbox("Primary Health Goal", HEALTH_GOALS)
    dietary_preference = st.selectbox("Dietary Preference", DIETARY_PREFERENCES)
    
    # Multiple selection for allergies
    allergies = st.multiselect("Allergies/Intolerances", ALLERGIES, default=["None"])
    
    # Free text inputs for food preferences
    food_preferences = st.text_area("Foods You Enjoy (comma separated)", height=100)
    food_dislikes = st.text_area("Foods You Dislike (comma separated)", height=100)

with col2:
    st.header("Your Nutrition Profile")
    
    # Calculate and display BMR and calorie needs
    if age and weight and height and gender:
        bmr = calculate_bmr(weight, height, age, gender)
        daily_calories = calculate_calorie_needs(bmr, activity_level)
        adjusted_calories = adjust_calories_for_goal(daily_calories, health_goal)
        
        st.subheader("Estimated Energy Needs")
        st.info(f"Basal Metabolic Rate (BMR): {bmr:.0f} calories/day")
        st.info(f"Daily Energy Needs: {daily_calories:.0f} calories/day")
        st.success(f"Adjusted for {health_goal}: {adjusted_calories:.0f} calories/day")
        
        # Simple macronutrient distribution
        if health_goal in ["Weight Loss", "Heart Health", "Diabetes Management"]:
            protein_pct = 30
            carb_pct = 40
            fat_pct = 30
        elif health_goal in ["Muscle Building", "Weight Gain"]:
            protein_pct = 35
            carb_pct = 45
            fat_pct = 20
        else:
            protein_pct = 25
            carb_pct = 50
            fat_pct = 25
        
        protein_g = (adjusted_calories * protein_pct/100) / 4  # 4 calories per gram
        carb_g = (adjusted_calories * carb_pct/100) / 4  # 4 calories per gram
        fat_g = (adjusted_calories * fat_pct/100) / 9  # 9 calories per gram
        
        st.subheader("Suggested Macronutrient Distribution")
        st.markdown(f"""
        - **Protein**: {protein_pct}% ({protein_g:.0f}g)
        - **Carbohydrates**: {carb_pct}% ({carb_g:.0f}g)
        - **Fats**: {fat_pct}% ({fat_g:.0f}g)
        """)
        
        # Display dietary notes
        st.subheader("Dietary Considerations")
        
        notes = []
        if "Vegetarian" in dietary_preference:
            notes.append("- Your vegetarian diet excludes meat but may include dairy and eggs.")
        elif "Vegan" in dietary_preference:
            notes.append("- Your vegan diet excludes all animal products. Ensure adequate B12, iron, and zinc intake.")
        
        for allergy in allergies:
            if allergy != "None":
                notes.append(f"- Excluding all sources of {allergy} from your meal plan.")
        
        if health_goal == "Weight Loss":
            notes.append("- Creating a moderate calorie deficit for sustainable weight loss.")
        elif health_goal in ["Weight Gain", "Muscle Building"]:
            notes.append("- Providing a calorie surplus for muscle growth and weight gain.")
        
        if not notes:
            notes.append("- No specific dietary restrictions identified.")
        
        for note in notes:
            st.markdown(note)
    
    # Add a button to get meal plan recommendations
    if st.button("Generate Personalized Meal Plan"):
        if not age or not weight or not height or not gender:
            st.error("Please fill in all required personal information.")
        else:
            recommendations, context = get_meal_plan_recommendations(
                age, gender, weight, height, activity_level,
                dietary_preference, health_goal,
                ", ".join(allergies) if allergies else "None",
                food_preferences, food_dislikes
            )
            
            st.header("Your Personalized Meal Plan")
            st.markdown(recommendations)
            
            # Show the source documents in an expander
            if context:
                with st.expander("View Nutritional References"):
                    for i, doc in enumerate(context):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(doc.page_content)
                        st.markdown("---")

# Add information about the app
st.markdown("---")
st.markdown("""
### About This Nutrition Assistant

This application uses AI and evidence-based nutrition science to create personalized meal plans tailored to your:

- Personal metrics (age, gender, weight, height)
- Activity level and exercise habits
- Health goals and dietary preferences
- Food allergies and intolerances
- Personal food preferences

The recommendations are based on:
1. Calculations of your individual energy needs
2. Scientific nutritional guidelines for your specific health goals
3. Evidence-based meal planning principles
4. Retrieval-augmented generation to provide accurate nutritional information

**Note**: While this tool provides evidence-based recommendations, it's not a substitute for professional medical advice. 
Always consult with a healthcare provider or registered dietitian before making significant changes to your diet, 
especially if you have pre-existing health conditions.
""")