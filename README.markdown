# AI-Powered Nutrition Assistant

The AI-Powered Nutrition Assistant is a web application designed to provide personalized, evidence-based meal plans tailored to a user's health goals, dietary preferences, allergies, and lifestyle. By integrating advanced AI techniques with nutritional science, the application calculates individual calorie needs and generates comprehensive 7-day meal plans, complete with shopping lists and practical implementation tips.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [License](#license)

## Features
- **Personalized Meal Plans**: Receive a 7-day meal plan tailored to your age, gender, weight, height, activity level, dietary preferences, allergies, and health goals.
- **Calorie and Macronutrient Calculations**: Uses the Mifflin-St Jeor equation to calculate Basal Metabolic Rate (BMR) and adjusts calorie needs based on activity level and health goals.
- **Dietary Customization**: Supports a wide range of dietary preferences (e.g., Vegetarian, Vegan, Keto) and excludes allergens or disliked foods.
- **RAG Pipeline Integration**: Utilizes Retrieval-Augmented Generation (RAG) with a nutrition knowledge base for accurate, evidence-based recommendations.
- **User-Friendly Interface**: Built with Streamlit for a clean and intuitive user experience.
- **Comprehensive Guidance**: Includes meal prep strategies, shopping lists, nutritional insights, and progress monitoring tips.

## Technologies Used
- **Streamlit**: Web application framework for building the user interface.
- **LangChain**: For implementing the RAG pipeline with embeddings and document retrieval.
- **FAISS**: Vector store for efficient similarity search of nutrition knowledge base.
- **HuggingFace Embeddings**: For generating text embeddings (model: `all-MiniLM-L6-v2`).
- **Grok (xAI)**: Language model for generating detailed meal plan recommendations.
- **PyPDF2**: For loading and processing PDF documents in the nutrition knowledge base.
- **Python**: Core programming language for calculations and logic.

## Installation
To run the AI-Powered Nutrition Assistant locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/nutrition-assistant.git
   cd nutrition-assistant
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Groq API key:
   ```plaintext
   GROQ_API_KEY=your-groq-api-key
   ```

5. **Prepare the Knowledge Base**:
   - Create a `nutrition_knowledge_base` directory in the project root.
   - Add PDF documents containing nutritional guidelines or evidence-based nutrition resources to this directory.

6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   The application will launch in your default web browser at `http://localhost:8501`.

## Usage
1. **Enter Personal Information**: Provide your age, gender, weight, height, and activity level.
2. **Specify Health Goals and Preferences**: Select your primary health goal, dietary preference, allergies, and list preferred or disliked foods.
3. **Generate Meal Plan**: Click the "Generate Personalized Meal Plan" button to receive a tailored 7-day meal plan.
4. **Explore Results**: View your estimated calorie needs, macronutrient distribution, dietary considerations, and detailed meal plan. Use the "View Nutritional References" expander to see source documents.

For best results, provide accurate personal metrics and detailed food preferences.

## Project Structure
```plaintext
nutrition-assistant/
├── app.py                    # Main Streamlit application
├── nutrition_knowledge_base/ # Directory for PDF documents (nutrition knowledge base)
├── .env                      # Environment variables (not tracked in Git)
├── requirements.txt          # Project dependencies
├── README.md                 # This file
```

## How It Works
1. **Calorie and Macronutrient Calculation**:
   - The application calculates Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation based on age, gender, weight, and height.
   - Daily calorie needs are determined by multiplying BMR by an activity level multiplier.
   - Calories are adjusted based on health goals (e.g., -500 for weight loss, +500 for weight gain).
   - Macronutrient distribution is tailored to the user's health goal.

2. **RAG Pipeline**:
   - PDF documents in the `nutrition_knowledge_base` directory are loaded and split into chunks using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.
   - Text embeddings are generated using HuggingFace's `all-MiniLM-L6-v2` model and stored in a FAISS vector store.
   - The RAG pipeline retrieves relevant documents and passes them to the Groq language model (`Llama3-8b-8192`) for generating detailed meal plans.

3. **Meal Plan Recommendations**:
   - A comprehensive prompt template guides the language model to create a 7-day meal plan, incorporating dietary preferences, allergies, and food preferences.
   - The output includes macronutrient targets, a detailed meal plan, shopping list, implementation tips, and nutritional insights.
   - Recommendations are displayed in a user-friendly format, with source documents available for transparency.

## Dependencies
The project requires the following Python packages (listed in `requirements.txt`):
```plaintext
streamlit==1.38.0
langchain-groq==0.2.0
langchain-community==0.3.1
faiss-cpu==1.9.0
huggingface_hub==0.25.1
sentence-transformers==3.1.1
PyPDF2==3.0.1
python-dotenv==1.0.1
```

Install them using:
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project root with the following:
```plaintext
GROQ_API_KEY=your-groq-api-key
```


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
