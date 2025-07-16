# utils/color_analyzer.py
import stone
import os
import tempfile
from PIL import Image
import io

def analyze_skin_tone(image_bytes):
    """
    Analyze the skin tone from an uploaded image.
    
    Args:
        image_bytes: The bytes of the uploaded image
        
    Returns:
        dict: A dictionary containing skin tone information
    """
    # Create a temporary file to store the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name
    
    try:
        # Process the image with STONE
        result = stone.process(
            temp_file_path,
            image_type="auto",
            return_report_image=True
        )
        
        # Extract the skin tone information
        if result and "faces" in result and len(result["faces"]) > 0:
            face_data = result["faces"][0]
            skin_info = {
                "hex_code": face_data.get("skin_tone", ""),
                "tone_label": face_data.get("tone_label", ""),
                "accuracy": face_data.get("accuracy", 0),
                "dominant_colors": face_data.get("dominant_colors", [])
            }
            
            # Get the report image for the first face
            report_images = result.get("report_images", {})
            report_image = None
            if report_images and 1 in report_images:
                report_image = report_images[1]
            
            return {"skin_info": skin_info, "report_image": report_image}
        else:
            return {"error": "No faces detected in the image"}
    
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_color_description(hex_code, color_type):
    """
    Get a textual description of a color based on its hex code and type.
    This is a simplified version - a more sophisticated approach would use
    a color classification algorithm.
    
    Args:
        hex_code: The hex code of the color
        color_type: The type of color (skin, hair, or eyes)
        
    Returns:
        str: A description of the color
    """
    # This is a very simplified approach
    # A real implementation would use a more sophisticated color classification
    if not hex_code:
        return "unknown"
    
    # Convert hex to RGB
    hex_code = hex_code.lstrip('#')
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    
    # Very basic classification based on RGB values
    if color_type == "skin":
        if r > 200 and g > 160 and b > 140:
            return "fair"
        elif r > 160 and g > 120 and b > 100:
            return "medium"
        else:
            return "deep"
    elif color_type == "hair":
        if r > 150 and g > 100 and b > 50:
            return "blonde"
        elif r > 100 and g > 50 and b > 40:
            return "brown"
        elif r < 80 and g < 60 and b < 60:
            return "black"
        elif r > 150 and g < 100 and b < 80:
            return "red"
        else:
            return "brown"
    elif color_type == "eyes":
        if b > r and b > g:
            return "blue"
        elif g > r and g > b:
            return "green"
        elif r > g and r > b:
            return "brown"
        else:
            return "hazel"
    
    return "unknown"
