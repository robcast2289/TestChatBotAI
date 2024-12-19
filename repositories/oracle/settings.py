import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    db_name:str = "wdesa"
    db_user:str = "DG_PYTHON"
    db_pass:str = "honda2025"
    db_host:str = "wdesa.galileo.edu"
    db_port:int = 1521
    db_dns:str = "wdesa.galileo.edu/wdesa"
    min_conns:int = 5
    max_conns:int = 25
    incr_conns:int = 1
    pool:str = ""