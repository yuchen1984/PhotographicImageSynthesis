# ***************************************************************************************************
# label_colour_definitions.py
# Global Variables that defining the label colour mapping for major public and private datasets
# Classes are global, declare global variables with global
# ***************************************************************************************************
# Variable Declarations
global label_colors_lip
global label_colors_atr
global label_colors_abof
global label_colors_binary

# Number of classes for segmentation
# Includes the background, can be numbered from 1-20 or 0-19 : First is always the background
# Put 20 for LIP, 18 for ATR, 21 for PASCAL VOC 
class_mapping_atr_to_abof = [[0, 0], [1, 10], [2, 5], [3, 10], [4, 6], [5, 7], [6, 7], [7, 8], [8, 10], [9, 9], \
                            [10, 9], [11, 1], [12, 3], [13, 3], [14, 2], [15, 2], [16, 10], [17, 10]]
                            
class_mapping_lip_to_abof = [[0, 0], [1, 1], [2, 6], [3, 5], [4, 2], [5, 7], [6, 2], [7, 9], [8, 9], [9, 10], \
                            [10, 6], [11, 3], [12, 3], [13, 10], [14, 10], [15, 10], [16, 8], [17, 7], [18, 8], [19, 10]]

class_mapping_atr_to_binary_body = [[0, 0], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], \
                            [10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1], [16, 1], [17, 1]]

class_mapping_atr_to_binary_upper = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 1], [5, 0], [6, 0], [7, 1], [8, 0], [9, 0], \
                            [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]]

class_mapping_atr_to_binary_lower = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 1], [6, 1], [7, 0], [8, 0], [9, 0], \
                            [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]]

class_mapping_atr_to_binary_cantor = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 1], [5, 1], [6, 1], [7, 1], [8, 0], [9, 0], \
                            [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]]

class_mapping_atr_to_abof_collapse = [[0, 0], [1, 10], [2, 5], [3, 10], [4, 6], [5, 7], [6, 7], [7, 6], [8, 10], [9, 9], \
                            [10, 9], [11, 1], [12, 3], [13, 3], [14, 2], [15, 2], [16, 10], [17, 10]]


class_mapping_abof_to_binary_body = [[0, 0], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1]]
class_mapping_abof_to_binary_upper = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 1], [7, 0], [8, 1], [9, 0], [10, 0]]
class_mapping_abof_to_binary_lower = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 1], [8, 0], [9, 0], [10, 0]]
class_mapping_abof_to_binary_skirt = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 1], [8, 1], [9, 0], [10, 0]]
class_mapping_abof_to_binary_any_garment = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 1], [7, 1], [8, 1], [9, 0], [10, 0]]

                            
# Color map for LIP
label_colors_lip = [(0,0,0),    # [BLACK]               00 = background     
                 (128,0,0),     # [RED]                 01 = face           
                 (0,128,0),     # [GREEN]               02 = upper clothes  
                 (255,128,0),   # [ORANGE]              03 = hair           
                 (0,0,128),     # [BLUE]                04 = right arm      
                 (128,0,128),   # [PURPLE]              05 = pants          
                 (255,255,0),   # [YELLOW]              06 = left arm       
                 (128,128,128), # [GREY]                07 = right shoe     
                 (73,2,6),      # [BROWN]               08 = left shoe         
                 (239,222,205), # [ALMOND]              09 = hat            
                 (102,93,30),   # [BRONZE]              10 = coat           
                 (59,122,87),   # [AMAZON]              11 = right leg      
                 (161,202,241), # [BABY BLUE]           12 = left leg          
                 (255,145,175), # [PINK]                13 = gloves         
                 (159,129,112), # [BEAVER]              14 = socks          
                 (209,159,232), # [BRIGHT UBE]          15 = sunglasses     
                 (8,232,222),   # [TURQUOISE]           16 = dress          
                 (74,255,0),    # [CHROLOPHYLL GREEN]   17 = skirt          
                 (194,59,34),   # [PASTEL RED]          18 = jumpsuits      
                 (102,180,71)]  # [APPLE GREEN]         19 = scarf          
                 
# Color map for ATR
label_colors_atr = [(0,0,0),    # [BLACK]               00 = background     
                 (239,222,205), # [ALMOND]              01 = hat           
                 (255,128,0),   # [ORANGE]              02 = hair
                 (209,159,232), # [BRIGHT UBE]          03 = sunglasses           
                 (0,128,0),     # [GREEN]               04 = upper clothes      
                 (74,255,0),    # [CHROLOPHYLL GREEN]   05 = skirt          
                 (128,0,128),   # [PURPLE]              06 = pants       
                 (8,232,222),   # [TURQUOISE]           07 = dress     
                 (255,255,255), # [WHITE]               08 = belt       
                 (73,2,6),      # [BROWN]               09 = left shoe            
                 (128,128,128), # [GREY]                10 = right shoe           
                 (128,0,0),     # [RED]                 11 = face      
                 (161,202,241), # [BABY BLUE]           12 = left leg          
                 (59,122,87),   # [AMAZON]              13 = right leg         
                 (255,255,0),   # [YELLOW]              14 = left arm           
                 (0,0,128),     # [BLUE]                15 = right arm      
                 (208,255,20),  # [LIME]                16 = bag          
                 (102,180,71)]  # [APPLE GREEN]         17 = scarf  

                 
# Color map for ABOF (Metail data)
label_colors_abof = [(0,0,0),    # [BLACK]              00 = background     
                 (255,255,0),    # [YELLOW]             01 = face           
                 (255,128,0),    # [ORANGE]             02 = arms
                 (255,0,0),      # [RED]                03 = legs           
                 (178,255,102),  # [LIGHT GREEN]        04 = torso      
                 (255,0,127),    # [MAGENTA]            05 = hair          
                 (0,255,0),      # [GREEN]              06 = upper-body garments       
                 (0,0,255),      # [BLUE]               07 = lower-body garments     
                 (0,255,255),    # [CYAN]               08 = full-body garments       
                 (127,0,255),    # [PURPLE]             09 = shoes            
                 (160,160,160)]  # [GREY]               10 = accessories           


label_colors_binary = [(0,0,0),    # [BLACK]            00 = background     
                 (255,255,255)]    # [WHITE]            01 = foreground            


