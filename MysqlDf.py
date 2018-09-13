import pandas as pd
import  pymysql as PyMySQL


def GetMysqlDF(Tab, SQL):
        # Open database connection
        #db = PyMySQL.connect("localhost","testuser","test123","TESTDB" )
        #db = PyMySQL.connect("symdb.cuic64tea7ki.ap-south-1.rds.amazonaws.com","symdbadm","Swasti23", "BHAV")
        
        """
        cursor = Conn.cursor()
        
        # Execute SQL select statement
        cursor.execute("SELECT * FROM EQ_BHAV limit 2")
        # Get the number of rows in the resultset
        numrows = cursor.rowcount
        
        # Get and display one row at a time
        for x in range(0, numrows):
            row = cursor.fetchone()
            print (row[0], "-->", row[1])
        """
        print("Running extract for"+Tab)
        df= pd.read_sql(SQL,Conn)
        print(df)

        # Commit your changes if writing
        # In this case, we are only reading data
        # db.commit()
        # Close the connection
        Conn.close()

        return df;
