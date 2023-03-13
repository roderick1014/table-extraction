# table-extraction

#### **📝This program can process all the .pdf and image files in the folder and extract the table within them.**
#### **🔰Table Extraction Program by Roderick & Kevin for internship project in Capacura GmbH.**

<img src="https://user-images.githubusercontent.com/73574008/223568380-07135ddc-a63e-4be2-b560-902483a2eaeb.png"  width="80%" height="80%">

---

## Progress (2023/03/08)
- ✅ **Image/PDF preprocessing**
- ✅ **Hough line transformation**
- ✅ **Intersections processing**
- ✅ **Table formulation based on intersections**
- ✅ **Text recognition based on OCR**
- ✅ **Auto-rotation pages**
- ⬜️ **Short column line detection problem (which decreases the performance of the hough line transformation)**
- ⬜️ **Repeated header in consecutive pages**
- ⬜️ **Truncated texts in consecutive pages**
- ⬜️ **DPI & Hough line threshold parameters fine-tuning**

## Python Version
  * **3.7.10**

## Packages

* **numpy (1.21.6)**

* **pandas (1.3.5)**

* **tqdm (4.61.2)**

* **opencv-python (4.5.4.60)**

* **pdf2image (1.16.2)**
    * **[Tutorial](https://python.plainenglish.io/how-to-convert-a-pdf-to-jpg-with-python-but-explained-as-a-cooking-recipe-cc04c2044818)**
    * **Remember to add "poppler" to environmental variables. (Windows)**

*  **PyTesseract (5.3.0)**
    *  **[Tutorial 1](https://digi.bib.uni-mannheim.de/tesseract/)**
    *  **[Tutorial 2](https://vocus.cc/article/621cfdb3fd8978000162a2e8)**
    *   <details>
        <summary>MacOS Instructions</summary>
        
        ### 1. Install the python package for Tesseract
        ```sh
        pip3 install PyTesseract  
        ```
        ### 2. Install Tesseract
        ```sh
        sudo port install Tesseract    
        ```
        ### 3. Install the Tesseract package required by OCR and auto rotation
        ```sh
        sudo port install tesseract-eng
        sudo port install tesseract-deu
        sudo port install tesseract-osd
        ```
 
        ### 4. Set the TESSDATA_PREFIX environment variable to the Tesseract data directory
        ```sh
        export TESSDATA_PREFIX=/opt/local/share/tessdata/
        ```
        </details>  

## Instructions

* **To save each page in the PDF file, add --SAVE_EACH_PAGE:**
   ```sh
   python .\table_extraction.py --SAVE_EACH_PAGE
   ```

* **To draw and visualize the houghline & intersection dots in the image, add --DRAW:**
   ```sh
   python .\table_extraction.py --DRAW
   ```

* **To specify the threshold of the houghline, add --THRESHOLD 'number':**
   ```sh
   python .\table_extraction.py --THRESHOLD 1300
   ```

* **To specify the DPI parameter, add --DPI 'number':**
   ```sh
   python .\table_extraction.py --DPI 200
   ```
  
* **To specify the folder containing the pdf files:**
   ```sh
   python .\table_extraction.py --FILES_DIR ./
   ```
