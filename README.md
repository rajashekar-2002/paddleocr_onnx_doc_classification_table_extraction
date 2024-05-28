#Paddle OCR ONNX 

This project uses paddleOCR to detect and recognise text further this model is converted to ONNX format which can give faster inferance

1. **Clone the project**
   
   ```git clone https://github.com/rajashekar-2002/paddleocr_onnx_doc_classification_table_extraction.git ```


2. **Insatll python libraries**

   ``` pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' ```
    ```pip install -m requirements.txt```

3. **Add model to directory**

     https://drive.google.com/drive/folders/1LDWs3yP7D1C9RuBGIGdSCPSWkKYP9JFD?usp=sharing

4. **Perform OCR using paddleocr with ONNX**

    ```python paddle.py```

5. **Perform document structure recognition and table data extraction**

     ```pyhton allclassocr.py```

   This file recognises document sturcture and makes bonding boxes of class detected from list of Text, List, Figure, Table, Title after classification bounding boxes are cropped and given as an input to encopy.py [paddleocr in ONNX format] to detect text.

   If class Table is detected detect_table.py is called to extract text and print output in table format.

![print table](https://github.com/rajashekar-2002/paddleocr_onnx_doc_classification_table_extraction/blob/main/tableocr.png)

   note : table line detection is using opencv make sure given image is clear and has lines for table


