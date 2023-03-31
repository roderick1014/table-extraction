'''
    2023/03/08 - Table Extraction Program by Roderick & Kevin for internship project in Capacura GmbH.
'''

# Importing the necessary libraries.
import os
import re
import sys
import cv2
import math
import json
import os.path
import imutils
import itertools
import pytesseract
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import cmp_to_key
from collections import defaultdict
from argparse import ArgumentParser
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

# Defining the function to show images.
def img_show(img_array, height = 1000, name = 'window'):  # height = 1200
    cv2.imshow(name, imutils.resize(img_array, height))
    cv2.waitKey(0)

# Defining the function to read PDF files.
def read_pdf(file_name):
    # Reading the PDF file.
    pages = convert_from_path(file_name, args.DPI)
    return pages

# Remove table in the image
def remove_table(img, lines):
    # The table received by the `keep_table()` function might containing some non-table regions
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # The boolean map dilated_table_map indicates the pixels that the lines found by Hough transform go through.
    dilated_table_map = np.zeros_like(img, dtype=np.uint8)
    draw_line(dilated_table_map, lines)
    dilated_table_map = dilated_table_map.max(-1)
    dilated_table_map = cv2.dilate(dilated_table_map, dilation_kernel, iterations=6) != 0

    img[dilated_table_map] = np.ones((1, 1, 3), dtype=np.uint8) * 255

    if args.DRAW or args.DEBUG:
        img_show(dilated_table_map.astype(np.uint8)*255)
    return img

# Defining the function to process PDF files.
def process_pdf(path, rm_table=True):

    # Creating the folder.
    new_path_name = path[:len(path) - 4]
    mkdir(new_path_name)

    # Reading the PDF file.
    pages = read_pdf(path)

    # Defining a progress bar for the number of pages
    pages_bar = tqdm(range(len(pages)))

    # Initializing a table record.
    table_record = []

    # Processing each page of the PDF file.
    for idx in pages_bar:
        # if idx < 2:
        #     continue
        pages_bar.set_description_str(' * Filename "' + path + '"')
        pages_bar.set_postfix(page = idx + 1)

        # Saving each page as an image.
        if args.SAVE_EACH_PAGE:
            pages[idx].save(new_path_name + '/page_' + str(idx) + '.jpg', 'JPEG')

        # Converting the page to a NumPy array.
        img_array = pil2np(pages[idx])

        # Auto-rotating the page.
        img_array = auto_rotation(img_array)

        # Creating a copy of the image array for later.
        org_img = img_array.copy()

        # Displaying the image if the DEBUG or DRAW mode is on.
        if args.DRAW or args.DEBUG:
            img_show(img_array)

        img_array[img_array<225] = 0
        if args.DRAW or args.DEBUG:
            img_show(img_array)

        # Getting the dimensions of the image.
        h, w, _ = img_array.shape

        # Binarizing the image.
        binarized_img_array = canny_img(img_array)

        # Displaying the image if the DEBUG or DRAW mode is on.
        if args.DRAW or args.DEBUG:
            img_show(binarized_img_array)

        # Removing the text from the image.
        rm_txt_img = remove_text(img_array, binarized_img_array)

        # Displaying the image if the DEBUG or DRAW mode is on.
        if args.DRAW or args.DEBUG:
            img_show(rm_txt_img)                # DEBUG

        # Keeping only the table from the image.
        binarized_img_array = keep_table(rm_txt_img)

        # Displaying the image if the DEBUG or DRAW mode is on.
        if args.DRAW or args.DEBUG:
            img_show(binarized_img_array)

        # Getting the lines using the Hough Line Transform.
        lines = houghline(binarized_img_array)

        # If lines were not detected, move to the next page.
        if lines is not None:

            if args.DRAW or args.DEBUG:
                draw_line(img_array, lines)
                img_show(img_array)

            # Getting the intersections of the lines.
            intersections = process_intersections(lines, h, w)

            # If there are no intersections, move to the next page.
            if len(intersections) == 0:
                continue

            # Displaying the image if the DEBUG or DRAW mode is on.
            if args.DRAW or args.DEBUG:
                img_array = draw_intersections(img_array, intersections)
                img_show(img_array)
            
            # Remove table in the image
            if rm_table:
                org_img = remove_table(org_img, lines)

            table_record.append(text_extraction_faster(org_img, intersections))

    # Table concatenation
    table_processing(table_record, new_path_name)

    print('='*120)

# Defining the function to process and save the table.
def table_processing(table_record, new_path_name):
    '''
        This function takes in a list of tables and saves them in a JSON file.
        It first concatenates all the tables horizontally and then transposes them to make them vertical.
        After that, it converts the table into a Pandas DataFrame and then to a dictionary which is then saved as a JSON file.
    '''

    table = np.concatenate(table_record, axis = 1)
    table = np.transpose(table, (1, 0))

    df = pd.DataFrame(table[1:], columns = table[0])
    table_dict = df.to_dict('records')

    js_table = json.dumps(table_dict, indent = 4, ensure_ascii=False).encode('utf8')
    js_file = open(new_path_name + '.json', 'wb')
    js_file.write(js_table)
    js_file.close()

    if args.DEBUG:
        print(table)
        print(len(table))
        print(len(table[0]))

# Defining the function to conduct canny edge detection.
def canny_img(img_array):
    '''
        This function applies the Canny edge detection algorithm to an image to detect edges.
    '''
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = cv2.Canny(img_array, 100, 200, apertureSize=3)

    return img_array

# Defining the function to remove the text and keep the table for further processing.
def keep_table(img, dilation_textRegion=True, filter_noise=True):
    '''
        This function applies morphological operations to an image to extract the table.
        It first converts the image to grayscale and applies a threshold.
        It then applies erosion and dilation operations to extract horizontal and vertical lines from the image.
        These lines are then combined to create a mask for the table.
    '''

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    img_bin = 255 - img_bin

    if args.DRAW or args.DEBUG:
        img_show(img_bin)

    if np.sum(img_bin) != 0:

        # cv2.waitKey(0)
        img_bin1 = 255 - img
        thresh1, img_bin1_otsu = cv2.threshold(img_bin1,128,255,cv2.THRESH_OTSU)

        img_bin2 = 255 - img
        thresh1, img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 100))
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)   # iterations 3 / 18

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 100, 1))
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=1)
        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=3)   # iterations 3

        vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)

        if args.DRAW or args.DEBUG:
            img_show(vertical_horizontal_lines)

        thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 126, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        """
        1. Solve the underline problem:
            - Problem: The words with underlines may be detected as lines, because underlines are straight and continious.
            - Solution: Since the underlines are always close to the words, 
                        applying dilation to the removed regions can clear the underlines while still preserving most of the table lines.

        2. Solve the noise problem:
            - Problem: The PDF files generated by the scanner sometimes contain noise, which is also considered a removed region. 
                       To solve the underline problem, we remove pixels that are too close to the removed regions. 
                       This process also removes some table lines that are near the noisy regions.
            - Solution: Since the noise is relatively small, so the noise can be removed by the opening operation (a erosion operation followed by a dilation operation)
        """
        if dilation_textRegion:
            smaller_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # 1.1. After removing the table lines, `img_bin_otsu` will be the removed regions
            vertical_horizontal_lines_dilated = vertical_horizontal_lines.copy()
            vertical_horizontal_lines_dilated = cv2.dilate(vertical_horizontal_lines_dilated, smaller_dilation_kernel, iterations=3)
            img_bin_otsu[vertical_horizontal_lines_dilated==255] = 0
            if args.DRAW or args.DEBUG:
                img_show(img_bin_otsu)
            
            # 2. Filter out noise if we set the `filter_noise` flag 
            if filter_noise:
                kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                if args.DRAW or args.DEBUG:
                    img_show(img_bin_otsu)
                img_bin_otsu = cv2.erode(img_bin_otsu, kernel_noise, iterations=2)
                if args.DRAW or args.DEBUG:
                    img_show(img_bin_otsu)
                img_bin_otsu = cv2.dilate(img_bin_otsu, kernel_noise, iterations=1)
                if args.DRAW or args.DEBUG:
                    img_show(img_bin_otsu)

            # 1.2. Apply the dilation operation to the removed regions
            larger_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
            img_bin_otsu = cv2.dilate(img_bin_otsu, larger_dilation_kernel, iterations=3)
            if args.DRAW or args.DEBUG:
                img_show(img_bin_otsu)

            # 1.3. Remove the regions near to the removed regions by the dilated removed region map (`img_bin_otsu`)
            vertical_horizontal_lines[img_bin_otsu==255] = 0
            if args.DRAW or args.DEBUG:
                img_show(vertical_horizontal_lines)

        return vertical_horizontal_lines
    else:
        return np.zeros_like(img_bin)

def find_minValidSquare(img_array):
    if (img_array == 255).sum() < 10:
        left_most_x = 0
        right_most_x = img_array.shape[1]
        top_most_y = 0
        bottom_most_y = img_array.shape[0]
    else:
        last_nonzero_idx_x = np.max(np.argwhere(img_array == 255), axis=1)
        last_nonzero_idx_y = np.max(np.argwhere(img_array == 255), axis=0)
        left_most_x = np.argmax(np.argmax(img_array, 0)!=0)
        right_most_x = max(last_nonzero_idx_x)
        top_most_y = np.argmax(np.argmax(img_array, 1)!=0)
        bottom_most_y = min(last_nonzero_idx_y)
    return ((left_most_x, top_most_y), (right_most_x, top_most_y), (right_most_x, bottom_most_y), (left_most_x, bottom_most_y)), bottom_most_y - top_most_y, right_most_x - left_most_x

# Eliminate the overlapped lines
def eliminate_lines(lines, threshold=25, mode='m'):
    '''
    This function eliminate the redundant lines by the following steps:
        1. It groups lines that are in close proximity to each other into the same cluster.
        2. For each cluster, it selects a representative line. 
           This can be done by calculating the median angle or position of the lines in the cluster, 
           or by identifying the line that is closest to the vertical or horizontal axis.
    '''
    
    # Step 1: Groups lines that are in close proximity to each other into the same cluster.
    combos = itertools.combinations(lines, 2)
    clustered_lines = set()
    clusters = dict()
    prev_line = lines[0]
    for line1, line2 in combos:
        if str(line1) != str(prev_line) and str(prev_line) not in clustered_lines:
            clustered_lines.add(str(prev_line))
            clusters[str(prev_line)].append(prev_line)
        prev_line = line1
        if str(line1) not in clusters.keys() and str(line1) not in clustered_lines:
            clusters[str(line1)] = list()
        if str(line1) in clustered_lines or str(line2) in clustered_lines:
            continue
        if abs(line1[0][0] - line2[0][0]) <= threshold:
            clustered_lines.add(str(line2))
            clusters[str(line1)].append(line2)
    if str(prev_line) not in clustered_lines:   
        clusters[str(prev_line)].append(prev_line)
    if str(lines[-1]) not in clustered_lines:   
        clusters[str(lines[-1])] = [lines[-1]]
    
    # Step 2: Determine the representative line for each cluster.
    lines_to_keep = list()
    for cluster in clusters.values():
        cluster_array = np.array(cluster)
        if mode == 'p':
            # Determine the final line as the line with the angle closest to 0 or 90 degree
            if abs(cluster_array[0][0][1] - math.radians(0)) < abs(cluster_array[0][0][1] - math.radians(90)):
                lines_to_keep.append(np.expand_dims(cluster_array[np.argmin(abs(cluster_array[:, 0, 1]-math.radians(0)))], 0))  
            else:
                lines_to_keep.append(np.expand_dims(cluster_array[np.argmin(abs(cluster_array[:, 0, 1]-math.radians(90)))], 0))  
        else:
            # Determine the final line as the line has a median value of rho
            lines_to_keep.append(np.expand_dims(cluster_array[np.argmax(cluster_array==np.median(cluster_array, 0)[0][0], 0)[0][0]], 0)) 
    lines_to_keep = np.concatenate(lines_to_keep, 0)
    return lines_to_keep

# Defining the function to conduct hough transform.
def houghline(img_array, eliminate_overlapped_lines=True, tolerable_shifted_angle=5):
    '''
        This function applies the Hough transform to an image to detect lines. It returns the detected lines.
        We use a dynamic threshold for both the vertical and the horizontal lines.
        The threshold is set depending on the width or the height of the minimum square containing all candidate points.
        Specifically, the threshold is equal to the width (or height) times a ratio, which we set it as `args.THRESHOLD_PERCENTAGE`.
    '''

    # Find the width and height of the minimum square region that containing all candidate edge points 
    corner_points, square_h, square_w = find_minValidSquare(img_array)

    # Draw the minimum square region that containing all candidate edge points 
    if args.DRAW or args.DEBUG:
        img_array_debug = np.array(np.broadcast_to(np.expand_dims(img_array, -1), (*img_array.shape, 3)))
        cv2.line(img_array_debug, corner_points[0], corner_points[1], (80, 128, 255), 2)
        cv2.line(img_array_debug, corner_points[1], corner_points[2], (80, 128, 255), 2)
        cv2.line(img_array_debug, corner_points[2], corner_points[3], (80, 128, 255), 2)
        cv2.line(img_array_debug, corner_points[3], corner_points[0], (80, 128, 255), 2)
        img_show(np.array(img_array_debug))

    rho, theta, thresh_per = 2, np.pi/180, args.THRESHOLD_PERCENTAGE
    lines_horizontal = cv2.HoughLines(img_array, rho, theta, int(square_w*thresh_per), min_theta=math.radians(90-tolerable_shifted_angle), max_theta=math.radians(90+tolerable_shifted_angle))
    # Many lines are detected for a line of the table. We use `eliminate_lines()` to remove the redundant lines.
    if lines_horizontal is not None and eliminate_overlapped_lines:
        lines_horizontal = eliminate_lines(lines_horizontal)
    lines_vertical = cv2.HoughLines(img_array, rho, theta, int(square_h*thresh_per), min_theta=math.radians(-tolerable_shifted_angle), max_theta=math.radians(tolerable_shifted_angle))
    if lines_vertical is not None and eliminate_overlapped_lines: 
        lines_vertical = eliminate_lines(lines_vertical)
    lines = None if lines_horizontal is None and lines_vertical is None else [lines_horizontal, lines_vertical]
    return lines


# Defining the function to draw the line.
def draw_line(img, lines):
    '''
        This function takes in an image and a set of lines and draws the lines on the image.
        If the line is vertical, it is drawn in red; otherwise, it is drawn in blue.
    '''
    lines = np.concatenate(lines, 0)
    if lines is not None:
        for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 3000 * (-b))
                    y1 = int(y0 + 3000 * (a))
                    x2 = int(x0 - 3000 * (-b))
                    y2 = int(y0 - 3000 * (a))
                    # If the radius is between 1.50 & 1.80 (85.94 ~ 103.13 deg), it's considered as a vertical line.
                    if 1.50 <= theta <= 1.80:
                        cv2.line(img,(x1, y1),(x2, y2), (80, 128, 255), 2)
                    else:
                        cv2.line(img,(x1, y1),(x2, y2),(255, 0, 0), 2)

# Defining the function to draw the intersection dots.
def draw_intersections(img, intersections):
    '''
        This function takes in an image and a set of points and draws circles at the points on the image.
    '''

    if intersections is not None:
        for intersection in intersections:
            if intersection is not None:
                x, y = intersection
                img = cv2.circle(img, (x,y), radius=0, color=(0, 0, 255), thickness=15)
        return img

# Defining the function to reverse sub-lists.
def reverse_sublist(lst,start,end):
    '''
        This function takes in a list and two indices and reverses the sublist between the indices.
    '''
    lst[start:end] = lst[start:end][::-1]
    return lst

class DrawTokens:
    def __init__(self, font_style = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.9, thickness=2):
        self.font_style = font_style
        self.font_scale = font_scale
        self.thickness = thickness
    def draw(self, text, h, w, x, y, color='red'):
        token_crop = cv2.putText( np.ones((h, w, 3), dtype=np.uint8)*255, 
                                  text, (x, y), self.font_style, 
                                  self.font_scale, (0, 0, 255) if color=='red' else (255, 0, 0), self.thickness, cv2.LINE_AA)
        return token_crop
    def get_token_size(self, text):    
        (text_w, text_h), _ = cv2.getTextSize(text, self.font_style, self.font_scale, self.thickness)
        return (text_w, text_h)

# Defining the function to extract the text information. (A quicker version)
def text_extraction_faster(img, sorted_intersections):
    '''
        This function takes in an image and a set of sorted intersections and extracts the text from the table.
        It first checks whether the table has been rotated slightly and adjusts the sorted intersections accordingly.
        It then iterates over the intersections to determine the length of each row and column.
        Once the length of each row and column is known, it extracts the text from the table by cropping each cell and running OCR on it.
    '''

    for idx in range(1, len(sorted_intersections)):            # Until we find a different y-axis value, we obtain the length of the row.
        current_point = sorted_intersections[idx][0]
        previous_point = sorted_intersections[idx - 1][0]

        if (previous_point + 30 < current_point) or (previous_point - 30 > current_point):
            break

    row_intersections = idx

    if args.DRAW or args.DEBUG:
        print('='*120)

    TokenDrawer = DrawTokens()
    columns = list()
    max_col_width = 0
    for idx in range(0, len(sorted_intersections)-row_intersections, row_intersections):
        x0, y0 = sorted_intersections[idx][0], sorted_intersections[idx][1]
        x1, y1 = sorted_intersections[idx + 2 * row_intersections - 1][0], sorted_intersections[idx + 2 * row_intersections - 1][1]
        cropped_img = img[y0:y1, x0:x1]
        sub_cropped_imgs = list()
        token_col = "SpecialToken"
        text_w, text_h = TokenDrawer.get_token_size(token_col)
        min_width = text_w + 20
        upper_bound_y = sorted_intersections[idx][1]
        for i in range(row_intersections-1):
            # 1. Get the 4 corners of a cell
            y_lt = sorted_intersections[idx+i][1]
            y_rt = sorted_intersections[idx+i+row_intersections][1]
            y_lb = sorted_intersections[idx+i+1][1]
            y_rb = sorted_intersections[idx+i+1+row_intersections][1]
            top_y = min(y_lt, y_rt)
            bottom_y = max(y_lb, y_rb)

            # 2. Crop the cell based on the 4 corners
            # If the width of cropped_img <= min_width, pad the edges
            if cropped_img.shape[1] <= min_width:
                white_bkg = np.ones((cropped_img.shape[0], min_width, 3), dtype=np.uint8)*255
                white_bkg[:, (min_width-cropped_img.shape[1])//2:(min_width-cropped_img.shape[1])//2+cropped_img.shape[1], :] = cropped_img
                cropped_img = white_bkg
            sub_cropped_img = cropped_img[top_y-upper_bound_y:bottom_y-upper_bound_y, :]

            # 3. Append a image crop
            sub_cropped_imgs.append(sub_cropped_img)

            # 4. Draw token_col
            token_pos_x = (sub_cropped_img.shape[1])//2 - text_w//2
            token_pos_y = (text_h+100)//2 + text_h//2
            token_crop = TokenDrawer.draw(token_col, text_h+100, sub_cropped_img.shape[1], token_pos_x, token_pos_y)

            # 5. Append token
            if i < row_intersections-2:
                sub_cropped_imgs.append(token_crop)

        # 6. Concatenate the crops in the same column, and seperating them by tokens 
        combined_cells_cols = np.concatenate(sub_cropped_imgs, 0)
        columns.append(combined_cells_cols)
        max_col_width = max(max_col_width, combined_cells_cols.shape[1])

    columns_to_concat = list()
    token_row = "TokenRow"
    # 7. Concatenate all columns
    for i, col in enumerate(columns):
        # 7.1. If a column is thiner than the thickest column, make it's width equal to `max_col_width`
        if col.shape[1] < max_col_width:
            white_bkg = np.ones((col.shape[0], max_col_width, 3), dtype=np.uint8)*255
            white_bkg[:col.shape[0], :col.shape[1], :] = col
            col = white_bkg
        columns_to_concat.append(col)

        if i < len(columns)-1:
            # 7.2 Generate a token crop to seperate every columns
            text_w, text_h = TokenDrawer.get_token_size(token_row)
            token_pos_x = max_col_width//2 - text_w//2
            token_pos_y = (text_h+100)//2 + text_h//2
            token_crop = TokenDrawer.draw(token_row, text_h+100, max_col_width, token_pos_x, token_pos_y, color='blue')
            columns_to_concat.append(token_crop)
    combined_cells = np.concatenate(columns_to_concat, 0)
        
    if args.DRAW or args.DEBUG:
        img_show(combined_cells)

    # To detect Deutsch, we have to specify lang = "deu"
    text = pytesseract.image_to_string(combined_cells, lang = "deu")

    # Replace the \n to space for further processing.
    new_text = text.replace('\n', ' ').replace('- ', '')

    table_record = list()
    column_records = new_text.split(token_row)
    for col_record in column_records:
        table_record.append(col_record.split(token_col))
    for i, col in enumerate(table_record):
        for j, cell in enumerate(col):
            table_record[i][j] = cell.strip()
    return np.array(table_record)

# Defining the function to extract the text information. (A quicker version)
def text_extraction_fast(img, sorted_intersections, dilated_table_map, remove_table=True):
    '''
        This function takes in an image and a set of sorted intersections and extracts the text from the table.
        It first checks whether the table has been rotated slightly and adjusts the sorted intersections accordingly.
        It then iterates over the intersections to determine the length of each row and column.
        Once the length of each row and column is known, it extracts the text from the table by cropping each cell and running OCR on it.
    '''
    if remove_table:
       img[dilated_table_map] = np.ones((1, 1, 3), dtype=np.uint8) * 255

    for idx in range(1, len(sorted_intersections)):            # Until we find a different y-axis value, we obtain the length of the row.
        current_point = sorted_intersections[idx][0]
        previous_point = sorted_intersections[idx - 1][0]

        if (previous_point + 30 < current_point) or (previous_point - 30 > current_point):
            break

    row_intersections = idx

    if args.DRAW or args.DEBUG:
        print('='*120)

    table_record = []
    column_record = []
    for idx in range(0, len(sorted_intersections)-row_intersections, row_intersections):
        x0, y0 = sorted_intersections[idx][0], sorted_intersections[idx][1]
        x1, y1 = sorted_intersections[idx + 2 * row_intersections - 1][0], sorted_intersections[idx + 2 * row_intersections - 1][1]
        cropped_img = img[y0:y1, x0:x1]
        sub_cropped_imgs = list()
        token = "SpecialToken"
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness=2
        (text_w, text_h), _ = cv2.getTextSize(token, font_style, font_scale, thickness)
        min_width = text_w + 20
        for i in range(row_intersections-1):
            upper_bound_y = sorted_intersections[idx][1]
            y_lt = sorted_intersections[idx+i][1]
            y_rt = sorted_intersections[idx+i+row_intersections][1]
            y_lb = sorted_intersections[idx+i+1][1]
            y_rb = sorted_intersections[idx+i+1+row_intersections][1]
            top_y = min(y_lt, y_rt)
            bottom_y = max(y_lb, y_rb)
            # If the width of cropped_img <= min_width, pad the edges
            if cropped_img.shape[1] <= min_width:
                white_bkg = np.ones((cropped_img.shape[0], min_width, 3), dtype=np.uint8)*255
                white_bkg[:, (min_width-cropped_img.shape[1])//2:(min_width-cropped_img.shape[1])//2+cropped_img.shape[1], :] = cropped_img
                cropped_img = white_bkg
            sub_cropped_img = cropped_img[top_y-upper_bound_y:bottom_y-upper_bound_y, :]
            sub_cropped_imgs.append(sub_cropped_img)
            token_crop = cv2.putText(
                            np.ones((text_h+100, sub_cropped_img.shape[1], 3), dtype=np.uint8)*255, 
                            token, ((sub_cropped_img.shape[1])//2-text_w//2, (text_h+100)//2+text_h//2), font_style, 
                            font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            if i < row_intersections-2:
                sub_cropped_imgs.append(token_crop)
        cropped_img = np.concatenate(sub_cropped_imgs, 0)
        if args.DRAW or args.DEBUG:
            img_show(cropped_img)

        # To detect Deutsch, we have to specify lang = "deu"
        text = pytesseract.image_to_string(cropped_img, lang = "deu")

        # Replace the \n to space for further processing.
        new_text = text.replace('\n', ' ').replace('- ', '')

        column_record = new_text.split(token)
        for i, record in enumerate(column_record):
            column_record[i] = record.strip()
        if len(column_record) == (row_intersections - 1):

            table_record.append(column_record)
            column_record = []

    return np.array(table_record)

# Defining the function to extract the text information.
def text_extraction_slow(img, sorted_intersections, dilated_table_map, remove_table=True):
    '''
        This function takes in an image and a set of sorted intersections and extracts the text from the table.
        It first checks whether the table has been rotated slightly and adjusts the sorted intersections accordingly.
        It then iterates over the intersections to determine the length of each row and column.
        Once the length of each row and column is known, it extracts the text from the table by cropping each cell and running OCR on it.
    '''
    if remove_table:
       img[dilated_table_map] = np.ones((1, 1, 3), dtype=np.uint8) * 255
    # if sorted_intersections[0][0] < sorted_intersections[1][0]:
    #     slight_right_rotated = True
    #     if args.DRAW or args.DEBUG:
    #         print("rotation happens!")
    # else:
    #     slight_right_rotated = False
        # If the table is rotate a little bit, the sorted intersection shold the inverse with the x-axis.
    for idx in range(1, len(sorted_intersections)):            # Until we find a different y-axis value, we obtain the length of the row.
        current_point = sorted_intersections[idx][0]
        previous_point = sorted_intersections[idx - 1][0]

        if (previous_point + 30 < current_point) or (previous_point - 30 > current_point):
            break

    row_intersections = idx
    # col_intersections = len(sorted_intersections) // row_intersections

    # if slight_right_rotated:
    #     for i in range(0, len(sorted_intersections), row_intersections):

    #         # if the table is rotate a little bit, the sorted intersection shold the inverse with the x-axis.
    #         if sorted_intersections[i][1] > sorted_intersections[i + 1][1]:
    #             sorted_intersections = reverse_sublist(sorted_intersections, i, i + row_intersections)

    if args.DRAW or args.DEBUG:
        print('='*120)

    table_record = []
    column_record = []
    for idx in range(len(sorted_intersections)):

        if (idx + row_intersections + 1) == len(sorted_intersections):
            break
        if ((idx + 1) % row_intersections) != 0:
            x0, y0 = sorted_intersections[idx][0], sorted_intersections[idx][1]
            x1, y1 = sorted_intersections[idx + row_intersections + 1][0], sorted_intersections[idx + row_intersections + 1][1]

            # Add/Substract values for enclosing the region.
            cropped_img = img[y0 : y1, x0 + 5 : x1 - 5]

            # To detect Deutsch, we have to specify lang = "deu"
            text = pytesseract.image_to_string(cropped_img, lang = "deu")

            # Replace the \n to space for further processing.
            new_text = text.replace('\n', ' ').replace('- ', '')

            column_record.append(new_text)
            if len(column_record) == (row_intersections - 1):

                table_record.append(column_record)
                column_record = []

    return np.array(table_record)

# Defining the function to remove the close intersection points.
def remove_close_points(input_list, threshold=(30, 30)):
    '''
        This function remove the close points and remove the redundant intersection dots.
    '''

    # Create all possible combinations of two points.
    combos = itertools.combinations(input_list, 2)
    # Create a list of the second point of each combination.

    # Iterate through each combination of two points and Check if the distance between two points is less than the threshold.
    points_to_remove = [point2
                        for point1, point2 in combos
                        if abs(point1[0] - point2[0]) <= threshold[0] and abs(point1[1] - point2[1]) <= threshold[1]]

    # Create a list of points that are not in the points_to_remove list.
    lines_to_keep = [point for point in input_list if point not in points_to_remove]

    # Return the lines_to_keep list.
    return lines_to_keep

# Defining the function to remove the texts in thetable.
def remove_text(img, binarized_img):
    '''
        This function takes in an image and a binary image, and applies morphological operations to remove any text in the image.
    '''

    # Create a rectangular kernel of size 15x3.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))

    # Perform morphological closing operation using the kernel.
    close = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Create a rectangular kernel of size 5x5.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # Perform morphological dilation operation using the kernel.
    dilate = cv2.dilate(close, dilate_kernel, iterations=2)

    # Find the contours in the dilated image
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the contours depending on the OpenCV version
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate through each contour and calculate the area of the contour, checking if the area of the contour is within the range.
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10 and area < 80000:
            # Get the bounding box of the contour
            x,y,w,h = cv2.boundingRect(c)
            # Draw a white rectangle over the contour
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), -1)

    # Return the modified image
    return img

def sorted_cmp(p1, p2):
    # If the two points are on the same horizontal line
    if abs(p1[0]-p2[0]) <= 30:
        if p1[1] > p2[1]:
            return 1
        elif p1[1] == p2[1]:
            return 0
        else:
            return -1
    else:
        if p1[0] > p2[0]:
            return 1
        else:
            return -1

# Defining the function to process intersections of lines generated by hough transform.
def process_intersections(lines, h, w):
    '''
        The process_intersections function takes in a list of lines detected in an image, and finds their intersections.
    '''

    # # Segment the lines based on angle using k-means.
    # segmented = segment_by_angle_kmeans(lines)

    # Find the intersections between the groups of lines.
    intersections = segmented_intersections(lines, h, w)

    if args.DRAW or args.DEBUG:
        print('Intersections: ', len(intersections))  # Print the number of intersections.

    # # Remove the close intersections.
    # intersections = remove_close_points(intersections)

    # Sort the intersections.
    # intersections = sorted(intersections, key=cmp_to_key(sorted_cmp))
    intersections = sorted(intersections, key=cmp_to_key(sorted_cmp))

    if args.DRAW or args.DEBUG:
        print('Refined Intersections: ', len(intersections))  # Print the number of intersections.

    return intersections

# Defining the function to find the intersection points.
def intersection(line1, line2, h, w):
    """
        Finds the intersection of two lines given in Hesse normal form.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    if w >= x0 >= 0 and h >= y0 >= 0:
        return [x0, y0]

# Defining the function to segment lines based on angle with k-means.
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
        This function groups lines based on angle with k-means.
        Uses k-means on the coordinates of the angle on the unit circle to segment 'k' angles inside 'lines'.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # Multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # Run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # Segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

# Defining the function to find the intersections between groups of lines.
def segmented_intersections(lines, h, w):
    '''
        This function finds the intersections between groups of lines.
    '''

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    # If theta is not too close (We hope lines are as vertical as well.)
                    if abs(line1[0][1] - line2[0][1]) > 0.6:
                        intersections.append(intersection(line1, line2, h, w))
    return intersections

# Defining the function to rotate the image/page automatically.
def auto_rotation(img):
    '''
        The auto_rotation function uses the Tesseract OCR library to determine the orientation of an image,
        and then rotates the image to align it with the horizontal axis.
    '''

    if args.DRAW or args.DEBUG:
        img_show(img)

    rot_data = pytesseract.image_to_osd(img)
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)
    angle = float(rot)

    # Rotate the image.
    if angle > 0:
        angle = 360 - angle
        rotated = rotate_bound(img, angle)
    elif angle == 0:
        rotated = img

    if args.DRAW or args.DEBUG:
        img_show(rotated)

    return rotated

# Defining the function to rotate the image/page.
def rotate_bound(img, angle):
    '''
        This function rotateds image base on warp and affine transformation.
    '''

    # Grab the dimensions of the image and then determine the center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine.
    # (the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))

# Defining the function to convert PIL to numpy array.
def pil2np(img):
    '''
        This function converts PIL to numpy array.
    '''
    return np.array(img)

# Defining the function to check if the path exists and make a folder for the output.
def mkdir(path):
    '''
        This function check the existence of the specified path and create the path.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        print(' - Path "'+ path +'" is created.')
    else:
        print(' - Path "'+ path +'" is already exitst, the program will overwrite the folder.')

# Defining the function to display the configuration.
def config_message():
    '''
        This function displays the configuration such as the mode (DRAW / DEBUG) and parameters such as DPI and THRESHOLD.
    '''

    print('='*120)
    print('DPI: ', args.DPI)
    print('THRESHOLD_PERCENTAGE: ', args.THRESHOLD_PERCENTAGE)
    print('SAVE_EACH_PAGE: ', args.SAVE_EACH_PAGE)
    print('DRAW: ', args.DRAW)
    print('DEBUG: ', args.DEBUG)
    print('='*120)

# Defining the main function.
def main():

    # List all the directions in the current folder.
    paths = [os.path.join(args.FILES_DIR, f) for f in os.listdir(args.FILES_DIR)]

    for path in paths:
        # If a pdf file is read.
        if path[-4:] == '.pdf':
            # Process the pdf file.
            process_pdf(path)
        # If an image file is read.
        if path[-4:] == '.tif' or 'tiff' or '.jpg' or '.png':
            pass

if __name__ == '__main__':

    # Parse the arguments.
    parser = ArgumentParser()
    parser.add_argument("--SAVE_EACH_PAGE", default=False, action='store_true')
    parser.add_argument("--DRAW", default=False, action='store_true')
    parser.add_argument("--DEBUG", default=False, action='store_true')
    parser.add_argument('--DPI', default = 200, type=int)  # DPI 200
    parser.add_argument('--THRESHOLD_PERCENTAGE', default = 0.8, type=float)
    parser.add_argument('--FILES_DIR', default = './', type=str)
    args = parser.parse_args()

    # Display the configuration.
    config_message()

    # Conduct main function.
    main()

    print('Operation finished.')

'''
    2023/03/08 - Table Extraction Program by Roderick & Kevin for internship project in Capacura GmbH.
'''