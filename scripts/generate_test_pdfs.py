
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def generate_table_pdf(filename="tests/samples/sample_table.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.drawString(100, height - 50, "Header Text")
    
    # Draw a table grid
    # 3 rows, 3 columns
    # x: 100, 200, 300, 400
    # y: 600, 550, 500, 450
    
    c.setLineWidth(1)
    
    # Horizontal lines
    for y in [600, 550, 500, 450]:
        c.line(100, y, 400, y)
        
    # Vertical lines
    for x in [100, 200, 300, 400]:
        c.line(x, 450, x, 600)
        
    # Add some text inside
    c.drawString(110, 570, "Row 1 Col 1")
    c.drawString(210, 570, "Row 1 Col 2")
    c.drawString(310, 570, "Row 1 Col 3")
    
    c.save()
    print(f"Generated {filename}")

def generate_textbox_pdf(filename="tests/samples/sample_textbox.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.drawString(50, height - 50, "Normal body text at the top.")
    
    # Draw a floating textbox
    # Rect: x=150, y=500, w=200, h=100
    # Filled background to trigger heuristics
    c.setFillColor(colors.lightgrey)
    c.rect(150, 500, 200, 100, fill=1, stroke=1)
    
    c.setFillColor(colors.black)
    c.drawString(160, 580, "This is a floating textbox.")
    c.drawString(160, 560, "It has a background and border.")
    
    c.drawString(50, 400, "Normal body text below the box.")
    
    c.save()
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_table_pdf()
    generate_textbox_pdf()
