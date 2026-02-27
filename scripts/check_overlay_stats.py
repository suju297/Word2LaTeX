from PIL import Image
import sys

def analyze_overlay(path):
    img = Image.open(path)
    pixels = list(img.getdata())
    total = len(pixels)
    
    # Red = (255, 0, 0) - Ref only
    # Cyan = (0, 255, 255) - Gen only
    # Black/Gray = Match
    # White = Background
    
    red_count = 0
    cyan_count = 0
    match_count = 0
    
    for r, g, b in pixels:
        if r > 200 and g < 50 and b < 50:
            red_count += 1
        elif r < 50 and g > 200 and b > 200:
            cyan_count += 1
        elif r < 50 and g < 50 and b < 50:
             match_count += 1
             
    print(f"Red Pixels (Missing in Gen): {red_count}")
    print(f"Cyan Pixels (Extra in Gen): {cyan_count}")
    print(f"Match Pixels: {match_count}")
    
    if red_count > 1000 and cyan_count < 100:
        print("CONCLUSION: MISSING_IN_GEN")
    elif red_count > 1000 and cyan_count > 1000:
        print("CONCLUSION: MOVED")
    else:
        print("CONCLUSION: MATCH")

if __name__ == "__main__":
    analyze_overlay(sys.argv[1])
