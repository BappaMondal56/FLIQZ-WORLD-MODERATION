from sqlalchemy import MetaData, Table
from database import engine

metadata = MetaData()

def get_dynamic_table(table_name: str):
    """
    Load ANY table dynamically at runtime (reflection).
    """
    return Table(table_name, metadata, autoload_with=engine)