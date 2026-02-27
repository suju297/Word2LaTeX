import fitz
import sys

def verify_images(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    if page_num >= doc.page_count:
        print(f"Error: Page {page_num} out of range (max {doc.page_count-1})")
        return

    page = doc[page_num]
    images = page.get_images(full=True)
    
    print(f"--- Analysis of {pdf_path} Page {page_num} ---")
    print(f"Total Images Found: {len(images)}")
    
    for idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        info = page.get_image_info(xrefs=True)
        
        # Match info to xref
        bbox = "Unknown"
        for item in info:
            if item['xref'] == xref:
                bbox = item['bbox']
                break
                
        print(f"Image {idx+1}:")
        print(f"  XREF: {xref}")
        print(f"  Format: {base_image['ext']}")
        print(f"  Size: {base_image['width']}x{base_image['height']}")
        print(f"  BBox on Page: {bbox}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_pdf_images.py <pdf_path> <page_num>")
        sys.exit(1)
        
    verify_images(sys.argv[1], int(sys.argv[2]))
