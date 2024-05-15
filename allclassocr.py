import cv2
import layoutparser as lp
from encopy import main as onnx_main
from PIL import Image
import pandas as pd
import time
from detect_table import table_detection
from tabulate import tabulate

def remove_lines(image):
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

def draw_box_using_text_blocks(text_blocks):
    x1 = int(text_blocks[0].block.x_1)
    y1 = int(text_blocks[0].block.y_1)
    x2 = int(text_blocks[0].block.x_2)
    y2 = int(text_blocks[0].block.y_2)
    crop_img = image[y1:y2, x1:x2]
    cv2_imshow(crop_img)
    return

def pandas_dataframe(data):
    print(data)
    #first = list(data.values())[0]
    first = data['cordinate'][0]
    #second = list(data.values())[1]
    #iter=0
    #first=first[0]
    print(first)
    #first = value[0]
    row = []
    my_list = []
    check_col = 0
    for key, value in data:
        #if abs(first - value) < abs(first-second + 5):
        if abs(first - value) < 5:
            my_list.append(key)
            #iter=iter+1
        else:
            first = value
            #second = list(data.values())[iter]
            #iter=iter+1
            row.append(my_list)
            my_list = []
            my_list.append(key)

    print(row)
    df = pd.DataFrame(row[1:], columns=row[0])
    print(df)
    return


def detect_text(image, block, ocr_class):


    table_text_list = []
    if ocr_class == "Table":
        print("================================================= Table ==============================================")

        # Extract block coordinates
        #x1 = int(block.block.x_1)
        #y1 = int(block.block.y_1)
        #x2 = int(block.block.x_2)
        #y2 = int(block.block.y_2)

        # Segment image using block coordinates with appropriate padding
        segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)

        # Call table_detection function for table-specific processing (implementation assumed)
        table_text_list = table_detection(segment_image)

        # Create pandas DataFrame, handling missing values
        df = pd.DataFrame(table_text_list)
        df = df.applymap(lambda x: '' if pd.isna(x) else x)

        # Generate table using tabulate
        table = tabulate(df, headers='firstrow', tablefmt='grid')

        # Print the extracted table
        print(table)

    else:
        # Handle non-table cases
        h, w = image.shape[:2]

        # Image segmentation (implementation assumed)
        segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)

        # Perform OCR using onnx_main (implementation assumed)
        data = onnx_main(segment_image)
        #print("------------------------------------------------------")
        #print(data['text'])

        output_string = data['text']

        if ocr_class == "Text":
            print("=============================================== Text ============================================")
            print(output_string[0])

        elif ocr_class == "List":
            print("=============================================== Text ============================================")
            print(output_string[0])

        elif ocr_class == "Figure":
            print("=============================================== Figure ============================================")
            if not output_string:
                print("------------no text detected in figure--------")
            else:
                print(output_string[0])

        elif ocr_class == "Title":
            print("======================================= Title ==================================")
            #print("Title data detected .. ")
            print(output_string[0])

    return  # No return value needed


def ocr_text(image, layout, ocr_class):
    text_blocks = lp.Layout([b for b in layout if b.type==ocr_class])
    print(text_blocks)
    #figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
    #print(figure_blocks)
    #text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
    #print(text_blocks)
    detect_text(image, text_blocks, ocr_class)

def main():
    start_time = time.time()
    image = cv2.imread("doc.png")
    image = image[..., ::-1]
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.81],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    table_model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_50_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
                                 label_map={0:"Table"})
    layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
    #print("++++++++++++++")
    #print(layout)
    #print("+++++++++++++++")
    layout1 = table_model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)

    if not layout and layout1:
        print("No text found ...")
        return

    for i in layout:
        print(i.type)


    filtered_blocks=[block for block in layout._blocks if block.type not in ["Figure","Table","List"]]
    all_blocks=filtered_blocks+layout1._blocks
    sorted_blocks=sorted(all_blocks,key=lambda block: block.block.y_1)
    #print("++++++++++++++")
    #print(sorted_blocks)
    #print("+++++++++++++++")
    for i in sorted_blocks:
        detect_text(image, i , i.type)


    end_time = time.time()
    time_difference = end_time - start_time
    print(time_difference)

    return







    all_list=[]
    for i in layout:
        all_list.append(i.type)

    unique_list = set(all_list)
    print("---------------------------------------------------------------")
    print(unique_list)
    try:
        unique_list.remove("Table")
        unique_list.remove("Figure")
    except:
        pass
    print(unique_list)
    unique_list = [x for x in unique_list]

    for i in unique_list:
        ocr_text(image, layout, i)

    print(layout1)


    if not layout1:
        print("No Table found ...")
        return
    #print("table layouout1 ======================== ============",layout1)
    ocr_text(image, layout1, "Table")


    end_time = time.time()
    time_difference = end_time - start_time
    print(time_difference)
if __name__ == "__main__":
    main()

