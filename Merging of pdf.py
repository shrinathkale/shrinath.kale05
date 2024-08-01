from PyPDF2 import PdfWriter
import os
merger = PdfWriter()
files = os.listdir()
for i in range(1,5):                      #for merging 1 to 4 PDF's
    for pdf in files:
        if(pdf.endswith(f"{i}.pdf")):
            merger.append(pdf)
merger.write("Merged-pdf.pdf")
merger.close()