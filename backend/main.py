from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import random
from config import db_config, apikey, access_key, secret_key
from pydantic import BaseModel
import logging
import requests
import boto3 # type: ignore
#from B5_Style_Sprout.camera import video_capture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
OVERWEAR_THRESHOLD = 10

# returns necessary info of a clothing item
def extract_info(clothing_item):
    if clothing_item == None: return None
    presigned_url = get_presigned_url(clothing_item["ImageUrl"])
    return {"Color": clothing_item["Color"],
            "ItemID": clothing_item["ItemID"],
            "URL": presigned_url,
            "ClothingType": clothing_item["ClothingType"],
            "UsageType": clothing_item["UsageType"],
            "NumUses": clothing_item["NumUses"]}

# gets the secure, presigned url from the s3 file
def get_presigned_url(file):
    s3 = boto3.client(
    's3',
    aws_access_key_id = access_key,
    aws_secret_access_key = secret_key,
    region_name = 'us-east-2'   
    )

    try:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': 'style-sprout',
                'Key': f'{file}.jpg'
            },
            ExpiresIn=3600  # url expires in 1 hour
        )
        return presigned_url
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# get the temperature of a location
def get_temperature(location):
    #if location == "Pittsburgh":
    #    return "cold"
    base_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={apikey}"
    response = requests.get(base_url)
    response.raise_for_status()

    data = response.json()
    lat = data[0]['lat']
    lon = data[0]['lon']

    base_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={apikey}&units=imperial"
    response = requests.get(base_url)
    response.raise_for_status()

    data = response.json()
    temperature = data['main']['temp']

    if temperature>=70:
        return 'hot'
    elif temperature>=55:
        return 'neutral'
    return 'cold'

# put one piece in top and have bottom empty
 
# cold
# jacket
# sweater, cardigan, blazer, hoodie
# top/bottom or one piece
# no shorts/tank tops

# neutral
# can have sweater, cardiagn, blazer, hoodie
# no jacket

# hot
# no sweater/jacket etc (no blazer)

# generate an outfit given a location and usage type
def fetch_outfit(location, usage_type):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        temp = get_temperature(location)
        
        cursor = connection.cursor(dictionary=True)

        if temp == "cold":
            top_clause = "AND ClothingType IN ('Tops', 'Tshirts')"
            bottom_clause = "AND ClothingType IN ('Trousers', 'Jeans', 'Leggings', 'Lounge Pants', 'Skirts')"
            overwear_clause = "AND ClothingType IN ('Sweaters', 'Hoodie', 'Cardigan')"
            jacket_clause = "AND ClothingType in ('Jackets')"
        elif temp == "neutral":
            top_clause = "AND ClothingType IN ('Tops', 'Tshirts', 'Tank')"
            bottom_clause = "AND ClothingType IN ('Trousers', 'Jeans', 'Leggings', 'Lounge Pants', 'Shorts', 'Skirts')"
            overwear_clause = "AND ClothingType IN ('Sweaters', 'Hoodie', 'Cardigan')"
            jacket_clause = ""
        else:  # hot
            top_clause = "AND ClothingType IN ('Tops', 'Tshirts', 'Tank')"
            bottom_clause = "AND ClothingType IN ('Trousers', 'Jeans', 'Leggings', 'Lounge Pants', 'Shorts', 'Skirts')"
            overwear_clause = ""  
            jacket_clause = ""

        # Prepare queries for fetching clean items with specified usage
        query_top = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1 {top_clause}
        """
        
        query_bottom = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1 {bottom_clause}
        AND ClothingType IN ('Jeans', 'Trousers', 'Skirts', 'Trackpants', 'Leggings', 'Shorts', 'Lounge Pants')
        """
        
        query_overwear = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1 {overwear_clause}
        """

        query_jacket = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1 {jacket_clause}
        """

        query_one_piece = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1 
        AND ClothingType IN ('Dresses', 'Jumpsuit')
        """

        query_blazer = f"""
        SELECT * FROM inventory 
        WHERE UsageType = '{usage_type}' AND Clean = 1
        AND ClothingType = 'Blazers'
        """
        
        # Execute queries
        cursor.execute(query_top)
        tops = cursor.fetchall()

        cursor.execute(query_bottom)
        bottoms = cursor.fetchall()

        cursor.execute(query_overwear)
        overwear = cursor.fetchall()

        cursor.execute(query_jacket)
        jackets = cursor.fetchall()

        cursor.execute(query_one_piece)
        one_pieces = cursor.fetchall()

        cursor.execute(query_blazer)
        blazers = cursor.fetchall()
        overwear_item = None
        jacket_item = None

        # no clean clothing that meet criteria
        if not ((tops and bottoms) or (one_pieces)):
            raise HTTPException(status_code=404, detail="No clean outfits available")

        # generate a blazer 40% of the time when it is formal, and not hot
        if temp != "hot" and blazers and usage_type == "formal" and random.random() < .4:
            overwear_item = random.choice(blazers)

        # generate overwear if we didn't generate a blazer
        if not overwear_item and overwear and random.random() < len(overwear)/OVERWEAR_THRESHOLD:
            overwear_item = random.choice(overwear)
        
        # generate jacket if it's cold
        if jackets and temp == "cold":
            jacket_item = random.choice(jackets)

        # 1 piece or 2 piece outfit
        if one_pieces and (random.random() < len(one_pieces) / (len(one_pieces) + min(len(tops), len(bottoms))) or not tops or not bottoms):
            # 1 piece
            # TODO: add user preferences here
            one_piece = random.choice(one_pieces)
            return {"top": extract_info(one_piece), 
                    "bottom": None, 
                    "overwear": extract_info(overwear_item), 
                    "jacket": extract_info(jacket_item)}
        else:
            # 2 piece 
            # TODO: add user preferences here
            top = random.choice(tops) if tops else None
            bottom = random.choice(bottoms) if bottoms else None
            return {"top": extract_info(top), 
                    "bottom": extract_info(bottom), 
                    "overwear": extract_info(overwear_item), 
                    "jacket": extract_info(jacket_item)}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# connect to the database
def create_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
        return None
    
# set all clothing to clean with 0 uses
def do_laundry():
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        reset_usage = """
        UPDATE inventory
        SET Clean = 1, NumUses = 0;
        """
        cursor.execute(reset_usage)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# select a generated outfit and update database table to reflect these
# items were selected
def select_outfit(primary, secondary, item_id1, item_id2, item_id3, item_id4):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        check_colors = f"""
        SELECT EXISTS (
            SELECT 1
            FROM user_preferences
            WHERE primary_color = '{primary}' AND secondary_color = '{secondary}'
        ) AS exists_flag;
        """
        cursor.execute(check_colors)
        exists = cursor.fetchall()
        
        if not exists[0]['exists_flag']:
            primary, secondary = secondary, primary

        update_preferences = f"""
        UPDATE user_preferences
        SET uses = uses + 1
        WHERE primary_color = '{primary}' AND secondary_color = '{secondary}'
        """
        cursor.execute(update_preferences)

        valid_ids = []
        for id in (item_id1, item_id2, item_id3, item_id4):
            if id != -1:
                valid_ids.append(id)
        valid_ids = tuple(valid_ids)

        update_usage = f"""
        UPDATE inventory
        SET NumUses = NumUses + 1
        WHERE ItemID in {valid_ids}
        """
        cursor.execute(update_usage)

        update_clean_status = f"""
        UPDATE inventory
        JOIN settings ON settings.ID = 1
        SET Clean = CASE
            WHEN NumUses >= settings.UsesBeforeDirty THEN 0
            WHEN NumUses < settings.UsesBeforeDirty THEN 1
        END;
        """
        cursor.execute(update_clean_status)

        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# edit the classifications of an item from the closet page
def edit_classification(id, usage, color, num_uses, item_type):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        update_classification = f"""
        UPDATE inventory
        SET ClothingType = '{item_type}',
        Color = '{color}',
        UsageType = '{usage}',
        NumUses = {num_uses}
        WHERE ItemID = {id}
        """
        
        cursor.execute(update_classification)

        update_clean_status = f"""
        UPDATE inventory
        JOIN settings ON settings.ID = 1
        SET Clean = CASE
            WHEN NumUses >= settings.UsesBeforeDirty THEN 0
            WHEN NumUses < settings.UsesBeforeDirty THEN 1
        END;
        """
        cursor.execute(update_clean_status)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# updates the number of uses for items to be considered "dirty"
def update_uses(uses):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        update_uses = f"""
        UPDATE settings
        SET UsesBeforeDirty = {uses}
        """
        cursor.execute(update_uses)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# adds a newly scanned item to the database
def add_item_to_db(item_info):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_last_id = f"""
        SELECT ItemID
        FROM inventory 
        ORDER BY ItemID desc 
        limit 1;
        """

        # TODO: change image_url to be set to whatever Riley names urls
        cursor.execute(get_last_id)
        last_id = cursor.fetchall()
        print(last_id)
        last_id = last_id[0]["ItemID"]
        image_url = f"username{last_id+1}"
        clothing_type = item_info["clothingType"]
        color = item_info["color"]
        usage_type = item_info["usageType"]
        
        add_item = f"""
        INSERT INTO inventory (ClothingType, Color, UsageType, ImageUrl)
        VALUES ('{clothing_type}', '{color}', '{usage_type}', '{image_url}');
        """
        cursor.execute(add_item)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# gets the image urls and ids on a page, also returns if there are more pages after
def closet_items(page):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_urls = f"""
        SELECT ImageUrl, ItemID,
        IF ({page* 6 + 6} < (SELECT COUNT(*) FROM inventory), 1, 0) 
        AS LastPage 
        FROM inventory 
        LIMIT 6 
        OFFSET {page* 6};
        """
        cursor.execute(get_urls)
        closet = cursor.fetchall()
        ids = []
        signed_urls = []
        for item in closet:
            signed_urls.append(get_presigned_url(item["ImageUrl"]))
            ids.append(str(item["ItemID"]))
        return {"urls": signed_urls, 
                "last_page": False if closet[0]["LastPage"] else True, 
                "ids": ids}
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# gets color, usage, clothing type, and number of use info from an item's id
def get_item_info(id):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_info = f"""
        SELECT Color, UsageType, ClothingType, NumUses 
        FROM inventory
        WHERE ItemID = {id}
        """
        cursor.execute(get_info)
        result = cursor.fetchall()
        logger.info(f"result: {result}")
        return result[0]
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# API routes

# get the outfit for a user given their location, and the usage type they request
@app.get("/outfit/{location}/{usage_type}")
def get_outfit(location: str, usage_type: str):
    valid_usage_types = {"Casual", "Formal"}

    if usage_type not in valid_usage_types:
        raise HTTPException(status_code=400, detail="Invalid weather/usage type")

    return fetch_outfit(location, usage_type)

# update the number of uses for an item to be considered "dirty"
@app.post("/laundry/update/{uses}")
def change_uses(uses: int):
    if uses<=0 or uses>100:
        raise HTTPException(status_code=400, detail="Invalid uses value")
    return update_uses(uses)

# update database table values when a user selects an outfit
@app.post("/select/{primary}/{secondary}/{item_id1}/{item_id2}/{item_id3}/{item_id4}")
def outfit_db_update(item_id1: int, item_id2: int, item_id3: int, item_id4: int, primary: str, secondary: str):
    return select_outfit(primary, secondary, item_id1, item_id2, item_id3, item_id4)

# updates the classifications of a clothing item
@app.post("/update/{id}/{usage}/{color}/{num_uses}/{item_type}")
def reclassify_closet(id: int, usage: str, color: str, num_uses: int, item_type: str):
    valid_usage_types = {"Casual", "Formal"}
    valid_color_types = {"Black", "Blue", "Brown", 
                         "Green", "Grey", "Orange", 
                         "Pink", "Purple", "Red", 
                         "White", "Yellow"}
    valid_item_types = {"Blazers", "Cardigan", "Dresses",
                        "Hoodie", "Jackets", "Jeans",
                        "Jumpsuit", "Leggings", "Lounge Pants",
                        "Shorts", "Skirts", "Sweaters", "Tank",
                        "Tops", "Trousers", "Tshirts"}
    if (id < 0 or num_uses < 0 or
        usage not in valid_usage_types or 
        color not in valid_color_types or 
        item_type not in valid_item_types):
        raise HTTPException(status_code=400, detail="Invalid classification change")
    return edit_classification(id, usage, color, num_uses, item_type)

# reset the clean status & number of uses for each item in the closet
@app.post("/laundry/reset")
def reset_laundry():
    return do_laundry()

# information about the outfit to store in the database
class Outfit_Info(BaseModel):
    clothingType: str
    color: str
    usageType: str

# POST request used by the Xavier NX to send information about the classified outfit to the database
@app.post("/outfit/info")
def change_uses(outfit_info: Outfit_Info):
    outfit_item_info = {
        "clothingType": outfit_info.clothingType,
        "color": outfit_info.color,
        "usageType": outfit_info.usageType,
    }

    add_item_to_db(outfit_item_info)
    return "Successful"

# returns image urls to display on closet page
@app.get("/closet_images/{page}")
def get_closet_images(page: int):
    if page < 0:
        raise HTTPException(status_code = 400, detail = "Invalid page value")
    return closet_items(page)

# returns labels of an item given its id
@app.get("/image_labels/{id}")
def get_image_labels(id: str):
    response = get_item_info(id)
    return {"labels": [response["Color"], response["ClothingType"],
            response["UsageType"], str(response["NumUses"])]}


# # Mock POST request used by the app to start scanning clothing 
# @app.post("/start/scanning")
# def start_scanning():
#     video_capture.user_start_scanning = True
#     return
