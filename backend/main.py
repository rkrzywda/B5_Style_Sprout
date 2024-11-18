from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import random
from config import db_config  

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
    cursor = connection.cursor(dictionary=True)
    reset_usage = f"""
    UPDATE inventory
    SET Clean = 1, NumUses = 0;
    """
    cursor.execute(reset_usage)
    return


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
        AND ClothingType IN ('Tshirt', 'Shirts', 'Sweatshirts', 'Tops')
        """
        
        cursor.execute(query_top, (usage_type,))
        tops = cursor.fetchall()

        query_bottom = f"""
        SELECT * FROM inventory 
        WHERE UsageType = %s AND Clean = 1 
        {season_clause} 
        AND ClothingType IN ('Jeans', 'Trackpants', 'Shorts', 'Trousers', 'Capris', 'Leggings')
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
    connection = create_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")
    try:
        cursor = connection.cursor(dictionary=True)
        return {}
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")
    finally:
        cursor.close()
        connection.close()

# API routes
@app.get("/outfit/{weather_type}/{usage_type}")
def get_outfit(weather_type: str, usage_type: str):
    valid_usage_types = ["casual", "formal", "sports"]
    valid_weather_types = ["warm", "cold", "neutral"]

    if weather_type not in valid_weather_types or usage_type not in valid_usage_types:
        raise HTTPException(status_code=400, detail="Invalid weather/usage type")

    return fetch_outfit(weather_type, usage_type)

@app.post("/clothing/info")
def change_uses(uses: int):
    if uses<=0:
        raise HTTPException(status_code=400, detail="Invalid uses value")
    return update_uses(uses)


#information about the outfit to store in the database
class Outfit_Info(BaseModel):
    clothingType: str
    color: str
    season: str
    usageType: str

#POST request used by the Xavier NX to send information about the classified outfit to the database
@app.post("/outfit/info")
def change_uses(outfit_info: Outfit_Info):
   
   outfit_item_info = {
       "clothingType": outfit_info.clothingType,
       "color": outfit_info.color,
       "season": outfit_info.season,
       "usageType": outfit_info.usageType,
   }
   return outfit_item_info

# view above post request on browser with
# http://ipaddress:8000/outfit/info
