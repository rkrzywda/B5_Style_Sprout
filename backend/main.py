from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import random
from config import db_config  
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

def fetch_outfit(weather_type, usage_type):
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)

        if weather_type == "warm":
            season_clause = "AND (Season = 'summer' OR Season = 'spring')"
        elif weather_type == "cold":
            season_clause = "AND (Season = 'fall' OR Season = 'winter')"
        else:
            season_clause = ""  

        query_top = f"""
        SELECT * FROM inventory 
        WHERE UsageType = %s AND Clean = 1 
        {season_clause} 
        AND ClothingType IN ('Tshirt', 'Shirts', 'Sweatshirts', 'Tops', 'Shirt')
        """
        
        cursor.execute(query_top, (usage_type,))
        tops = cursor.fetchall()

        query_bottom = f"""
        SELECT * FROM inventory 
        WHERE UsageType = %s AND Clean = 1 
        {season_clause} 
        AND ClothingType IN ('Jeans', 'Trackpants', 'Shorts', 'Trousers', 'Capris', 'Leggings', 'Skirt')
        """
        cursor.execute(query_bottom, (usage_type,))
        bottoms = cursor.fetchall()

        if not tops or not bottoms:
            raise HTTPException(status_code=404, detail="No clean outfits available for this weather/usage type")

        top = random.choice(tops)
        bottom = random.choice(bottoms)

        return {"top": top, "bottom": bottom}
    
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
        season = item_info["season"]
        usage_type = item_info["usageType"]
        
        add_item = f"""
        INSERT INTO inventory (ClothingType, Color, Season, UsageType, ImageUrl)
        VALUES ('{clothing_type}', '{color}', '{season}', '{usage_type}', '{image_url}');
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
    if (outfit_info.clothingType in ('Shorts')):
       season = 'Summer'
    elif (outfit_info.clothingType in ('Sweatshirts')):
        season = 'Winter'
    else:
        season = "Spring"
   
    outfit_item_info = {
        "clothingType": outfit_info.clothingType,
        "color": outfit_info.color,
        "season": season,
        "usageType": outfit_info.usageType,
    }

    add_item_to_db(outfit_item_info)

    return "Successful"

