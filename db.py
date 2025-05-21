import psycopg2

DB_HOST = 'localhost'
DB_PORT = '5433'
DB_NAME = 'pu2pay_v1'
DB_USER = 'postgres'
DB_PASSWORD = '12345'

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )