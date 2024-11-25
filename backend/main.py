from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import random
from config import db_config, apikey
from pydantic import BaseModel
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
OVERWEAR_THRESHOLD = 10

def get_temperature(location):
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
        blazer_item = None
        overwear_item = None

        # no clean clothing that meet criteria
        if not ((tops and bottoms) or (one_pieces)):
            raise HTTPException(status_code=404, detail="No clean outfits available")

        # generate a blazer 40% of the time when it is formal, and not hot
        if temp != "hot" and blazers and usage_type == "formal" and random.random() < .4:
            overwear_item = random.choice(blazers)

        # generate overwear if we didn't generate a blazer
        if not overwear_item and overwear and (temp == "cold" and random.random() < len(overwear)/OVERWEAR_THRESHOLD):
            overwear_item = random.choice(overwear)
        
        # generate jacket if it's cold
        if jackets and temp == "cold":
            jacket_item = random.choice(jackets)

        # 1 piece or 2 piece outfit
        if one_pieces and (random.random() < len(one_pieces) / (len(one_pieces) + min(len(tops), len(bottoms))) or not tops or not bottoms):
            # 1 piece
            # TODO: add user preferences here
            one_piece = random.choice(one_pieces)
            return {"top": one_piece, "bottom": None, "overwear": overwear_item, "jacket": jacket_item}
        else:
            # 2 piece 
            # TODO: add user preferences here
            top = random.choice(tops) if tops else None
            bottom = random.choice(bottoms) if bottoms else None
            return {"top": top, "bottom": bottom, "overwear": overwear_item, }
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()


def create_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None
    
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

def select_outfit(primary, secondary, item_id1, item_id2):
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

        update_usage = f"""
        UPDATE inventory
        SET NumUses = NumUses + 1
        WHERE ItemID = {item_id1} OR ItemID = {item_id2}
        """
        cursor.execute(update_usage)

        update_clean_status = f"""
        UPDATE inventory
        JOIN settings ON settings.ID = 1
        SET Clean = 0
        WHERE NumUses >= settings.UsesBeforeDirty;
        """
        cursor.execute(update_clean_status)

        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

def update_uses(uses):
    if uses<=0:
        raise HTTPException(status_code=404, detail="Uses cannot be 0")
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

# API routes
@app.get("/outfit/{weather_type}/{usage_type}")
def get_outfit(weather_type: str, usage_type: str):
    valid_usage_types = ["casual", "formal", "athletic"]
    valid_weather_types = ["warm", "cold", "neutral"]

    if weather_type not in valid_weather_types or usage_type not in valid_usage_types:
        raise HTTPException(status_code=400, detail="Invalid weather/usage type")

    return fetch_outfit(weather_type, usage_type)

@app.post("/laundry/update/{uses}")
def change_uses(uses: int):
    if uses<=0 or uses>100:
        raise HTTPException(status_code=400, detail="Invalid uses value")
    return update_uses(uses)

@app.post("/select/{primary}/{secondary}/{item_id1}/{item_id2}")
def outfit_db_update(item_id1: int, item_id2: int, primary: str, secondary: str):
    return select_outfit(primary, secondary, item_id1, item_id2)

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

