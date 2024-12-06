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

# deletes image from s3
def delete_image_from_s3(file):
    s3 = boto3.client(
    's3',
    aws_access_key_id = access_key,
    aws_secret_access_key = secret_key,
    region_name = 'us-east-2'   
    )
    
    try:
        response = s3.delete_object(Bucket='style-sprout', Key=f'{file}.jpg')
    except Exception as e:
        logger.info(f"While deleting {file}.jpg an error occurred: {e} ")

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
    #  return "cold"
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
        logger.info("hot")
        return 'hot'
    elif temperature>=55:
        logger.info("neutral")
        return 'neutral'
    logger.info("cold")
    return 'cold'

# check for location validity
def is_valid_location(location):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={apikey}"
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            return True
        else: return False
    except Exception as e:
        return False

# get user preferences
def get_user_preferences(cursor):
    preferences_query = """
    SELECT * FROM user_preferences
    """
    cursor.execute(preferences_query)
    user_preferences = cursor.fetchall()
    return user_preferences

# get outfit dislikes
def get_outfit_dislikes(cursor):
    dislikes_query = """
    SELECT * FROM OutfitDislikes
    """
    cursor.execute(dislikes_query)
    return cursor.fetchall()

# calculate weights for all top/bottom combinations based on user preferences
def get_weights(tops, bottoms, user_preferences, outfit_dislikes):
    weights = dict()
    dislikes_map = {(dislike["Top"], dislike["Bottom"]): dislike["Dislikes"] for dislike in outfit_dislikes}
    for top in tops:
        for bottom in bottoms:
            top_color = top["Color"]
            bottom_color = bottom["Color"]
            top_id = top["ItemID"]
            bottom_id = bottom["ItemID"]
            pref_weight = 1

            for pref in user_preferences:
                primary = pref["primary_color"]
                secondary = pref["secondary_color"]
                if ((primary == top_color and secondary == bottom_color) or
                    (primary == bottom_color and secondary == top_color)):
                    pref_weight = max(pref["uses"], 1)
                    break
            dislikes = dislikes_map.get((top_id, bottom_id), 0)
            weights[(top_id, bottom_id)] = max(pref_weight - dislikes, 0)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        for outfit in weights:
            weights[outfit] = 1
    return weights

# calculate weights for one piece outfits (dresses) based on user preferences
def get_weights_one_piece(tops, user_preferences, outfit_dislikes):
    weights = dict()
    dislikes_map = {dislike["Top"]: dislike["Dislikes"] for dislike in outfit_dislikes if dislike["Bottom"] is None}
    for top in tops:
        top_color = top["Color"]
        top_id = top["ItemID"]
        pref_weight = 1

        for pref in user_preferences:
            primary = pref["primary_color"]
            secondary = pref["secondary_color"]
            if (primary == top_color and secondary == top_color and pref["uses"] > 0):
                pref_weight = max(pref["uses"], 1)
                break
        dislikes = dislikes_map.get(top_id, 0)
        weights[top_id] = max(pref_weight - dislikes, 0)
    
    total_weight = sum(weights.values())
    if total_weight <= 0:
        for outfit in weights:
            weights[outfit] = 1
    return weights

# get the biased outfit based on weights with item ids as keys
def get_biased_outfit(weights):
    outfit_choices = list(weights.keys())
    outfit_weights = list(weights.values())

    generated_outfit = random.choices(outfit_choices, weights = outfit_weights, k = 1)[0]
    if type(generated_outfit) == tuple:
        return (get_item_from_id(generated_outfit[0]), get_item_from_id(generated_outfit[1]))
    return get_item_from_id(generated_outfit)

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
def fetch_outfit(usage_type):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)

        location_query = """
        SELECT Location
        FROM settings
        WHERE ID = 1
        """
        cursor.execute(location_query)
        location = cursor.fetchall()[0]["Location"]
        temp = get_temperature(location)

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
            if random.random() < .3:
                logger.info("Biased outfit")
                # biased outfit
                user_preferences = get_user_preferences(cursor)
                outfit_dislikes = get_outfit_dislikes(cursor)
                weights = get_weights_one_piece(one_pieces, user_preferences, outfit_dislikes)
                one_piece = get_biased_outfit(weights)
            else: 
                logger.info("Random outfit")
                # random outfit
                one_piece = random.choice(one_pieces)
            return {"top": extract_info(one_piece), 
                    "bottom": None, 
                    "overwear": extract_info(overwear_item), 
                    "jacket": extract_info(jacket_item)}
        else:
            # 2 piece 
            if random.random() < .3:
                logger.info("Biased outfit")
                # biased outfit
                user_preferences = get_user_preferences(cursor)
                outfit_dislikes = get_outfit_dislikes(cursor)
                weights = get_weights(tops, bottoms, user_preferences, outfit_dislikes)
                top, bottom = get_biased_outfit(weights)
            else:
                logger.info("Random outfit")
                # random outfit
                top = random.choice(tops) if tops else None
                bottom = random.choice(bottoms) if bottoms else None
            return {"top": extract_info(top), 
                    "bottom": extract_info(bottom), 
                    "overwear": extract_info(overwear_item), 
                    "jacket": extract_info(jacket_item)}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail="Database query failed HERE")
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
def update_settings(uses, location):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        update_uses = f"""
        UPDATE settings
        SET UsesBeforeDirty = {uses},
        Location = "{location}"
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
        image_url = item_info["imageName"]
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
        return result[0]
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

def get_item_from_id(id):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        logger.info(f"id: {id}")
        get_info = f"""
        SELECT ClothingType, Color, UsageType, Clean, NumUses, ItemID, ImageUrl 
        FROM inventory
        WHERE ItemID = {id}
        """
        cursor.execute(get_info)
        result = cursor.fetchall()
        return result[0]
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

def get_user_response():
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_info = f"""
        SELECT hasAccepted
        FROM privacy_notice
        """
        cursor.execute(get_info)
        result = cursor.fetchall()
        return result[0]
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

def accept_notice():
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_info = f"""
        UPDATE privacy_notice 
        SET hasAccepted = 1 
        WHERE id = 1
        """
        cursor.execute(get_info)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

def update_dislikes(item_id1, item_id2):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        top = item_id1
        bottom = item_id2 if item_id2 != -1 else "NULL"
        get_info = f"""
        INSERT INTO OutfitDislikes (Top, Bottom, Dislikes)
        VALUES ({top}, {bottom}, 1)
        ON DUPLICATE KEY UPDATE Dislikes = Dislikes + 1;
        """
        cursor.execute(get_info)
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()
    
def delete_item_from_id(item_id):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        get_url = f"""
        SELECT ImageUrl
        FROM inventory
        WHERE ItemID = {item_id}
        """
        cursor.execute(get_url)
        url = cursor.fetchall()[0]["ImageUrl"]
        delete_image_from_s3(url)
        delete_query = f"""
        DELETE FROM inventory
        WHERE ItemID = {item_id}
        """
        cursor.execute(delete_query)
        connection.commit()
        
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# API routes

# get the outfit for a user given their location, and the usage type they request
@app.get("/outfit/{usage_type}")
def get_outfit(usage_type: str):
    valid_usage_types = {"Casual", "Formal"}

    if usage_type not in valid_usage_types:
        raise HTTPException(status_code=400, detail="Invalid usage type")

    return fetch_outfit(usage_type)

# update the number of uses for an item to be considered "dirty" and user's location
@app.post("/settings/update/{uses}/{location}")
def change_uses(uses: int, location: str):
    if uses<=0 or uses>100 or not is_valid_location(location):
        raise HTTPException(status_code=400, detail="Invalid uses/location")
    return update_settings(uses, location)

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
    imageName: str

# POST request used by the Xavier NX to send information about the classified outfit to the database
@app.post("/outfit/info")
def change_uses(outfit_info: Outfit_Info):
    outfit_item_info = {
        "clothingType": outfit_info.clothingType,
        "color": outfit_info.color,
        "usageType": outfit_info.usageType,
        "imageName": outfit_info.imageName,
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

# returns if the user has agreed to our privacy notice
@app.get("/privacy_notice/")
def privacy_notice():
    response = get_user_response()
    return response["hasAccepted"]

# returns if the user has agreed to our privacy notice
@app.post("/privacy_notice/accept")
def accept_privacy_notice():
    accept_notice()

# dislike the given outfit
@app.post("/dislike/{itemid_1}/{itemid_2}")
def dislike_outfit(itemid_1: int, itemid_2: int):
    return update_dislikes(itemid_1, itemid_2)

# delete item from database and s3
@app.post("/delete/{itemid_1}")
def delete_item(itemid_1: int):
    return delete_item_from_id(itemid_1)
    

# # Mock POST request used by the app to start scanning clothing 
# @app.post("/start/scanning")
# def start_scanning():
#     video_capture.user_start_scanning = True
#     return
