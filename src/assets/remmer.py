from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    """
    Remove background from an image using rembg
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image (background removed)
    """
    # Open and process image
    input_image = Image.open(input_path)
    
    # Remove background
    output_image = remove(input_image)
    
    # Save result
    output_image.save(output_path)
    print(f"Background removed! Image saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    remove_background("/home/kami/Desktop/codebase/EagleLegal/src/assets/eagle.png", "output.png")