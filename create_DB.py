from sqlalchemy import create_engine
from sqlalchemy.sql import text

user = 'postgres'
password = 'postgres'
host = 'localhost'
port = 5433
database = 'postgres'

def get_connection():
    return create_engine(
        url="postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )

def create_table(engine):
    with engine.connect() as conn:
        conn.execute(text('''CREATE TABLE IF NOT EXISTS credit (
            BAD INTEGER,
            LOAN INTEGER,
            MORTDUE INTEGER,
            VALUE INTEGER,
            REASON VARCHAR(255),
            JOB VARCHAR(255),
            YOJ DOUBLE PRECISION,
            DEROG INTEGER,
            DELINQ INTEGER,
            CLAGE DOUBLE PRECISION,
            NINQ INTEGER,
            CLNO INTEGER,
            DEBTINC DOUBLE PRECISION
        )'''))

if __name__ == '__main__':
    try:
        # Get the connection object (engine) for the database
        engine = get_connection()
        create_table(engine)
        print(f"Connection to the {host} for user {user} created successfully.")
    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)
