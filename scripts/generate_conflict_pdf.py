
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def generate_conflict_pdf(filename="tests/samples/conflict_case.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Text that OOXML will see as a normal paragraph
    # But we will draw a box around it to trick heuristics
    
    c.drawString(100, 700, "Normal Paragraph Text")
    c.drawString(100, 680, "This text is in a box.")
    
    # Draw a box that looks like a textbox (filled)
    c.setFillColor(colors.lightgrey)
    # Rect around the text
    c.rect(90, 670, 200, 50, fill=1, stroke=1)
    
    c.setFillColor(colors.black)
    c.drawString(100, 690, "Text inside the box context.")
    
    c.save()
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_conflict_pdf()
