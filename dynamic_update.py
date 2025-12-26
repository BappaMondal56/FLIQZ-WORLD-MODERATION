from datetime import datetime
from sqlalchemy import select, update, insert
from database import get_db
from dynamic_table_loader import get_dynamic_table

def dynamic_update(payload: dict, animal_detected=False, das_detected=False, minor_detected=False, personal_info_detected=False, nsfw_detected=False, violence_detected=False, weapon_detected=False):
    """
    Generic UPSERT based on table_name, primary_key, key_value.
    Works for ANY table.
    """
    table_name = payload["table_name"]
    pk_name = payload["primary_key"]
    pk_value = payload["key_value"]

    # Dynamically load table
    table = get_dynamic_table(table_name)

    db = next(get_db())
    now = datetime.now()

    try:
        # 1. Check if row exists
        stmt = select(table).where(table.c[pk_name] == pk_value)
        existing = db.execute(stmt).fetchone()

        # 2. If row exists → UPDATE
        if existing:
            update_data = {
                k: v for k, v in payload.items()
                if k not in ["table_name", "primary_key", "key_value"]
                and k in table.c
            }
            if "updated_at" in table.c:
                update_data["updated_at"] = now

            # Add moderation flags to update_data if columns exist
            if "animal_detected" in table.c:
                update_data["animal_detected"] = 1 if animal_detected else 0

            if "is_das_detected" in table.c:
                update_data["is_das_detected"] = 1 if das_detected else 0

            if "minor_detected" in table.c:
                update_data["minor_detected"] = 1 if minor_detected else 0

            if "is_personal_details_detected" in table.c:
                update_data["is_personal_details_detected"] = 1 if personal_info_detected else 0   

            if "nsfw_detected" in table.c:
                update_data["nsfw_detected"] = 1 if nsfw_detected else 0

            if "violance_detected" in table.c:
                update_data["violance_detected"] = 1 if violence_detected else 0

            if "is_weapon_detected" in table.c:
                update_data["is_weapon_detected"] = 1 if weapon_detected else 0      
    

            stmt = (
                update(table)
                .where(table.c[pk_name] == pk_value)
                .values(update_data)
            )

            db.execute(stmt)
            db.commit()
            return True, "updated"
        
        else:
            return False, "row_not_found"

    

    except Exception as e:
        db.rollback()
        return False, str(e)