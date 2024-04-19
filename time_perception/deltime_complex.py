import csv

import pymysql

conn = pymysql.connect(host='127.0.0.1',
                     port=3306,
                     user='root',
                     password='241033',
                     db='datongresult',charset='utf8')
cursor=conn.cursor()

with open('datacsvdatong.csv', 'r',encoding="utf-8") as f:
    timeone=0
    timetwo=0
    timethree=0
    timefour=0
    timefive=0
    timesix=0
    timeseven=0
    timeeight=0
    timenine=0
    timeten=0
    timeeleven=0
    timetwelve=0
    reader = csv.reader(f)
    writer = csv.writer(f)
    for row in reader:
        id = row[0]
        time = row[2]
        season=""
        timelist=time.split("/")
        if timelist[1] == "1":
            season="1"
            timeone+=1
        if timelist[1] == "2":
            season = "1"
            timetwo+=1
        if timelist[1] == "3":
            season = "1"
            timethree+=1
        if timelist[1] == "4":
            season = "2"
            timefour+=1
        if timelist[1] == "5":
            season = "2"
            timefive+=1
        if timelist[1] == "6":
            season = "2"
            timesix+=1
        if timelist[1] == "7":
            season = "3"
            timeseven+=1
        if timelist[1] == "8":
            season = "3"
            timeeight+=1
        if timelist[1] == "9":
            season = "3"
            timenine+=1
        if timelist[1] == "10":
            season = "4"
            timeten+=1
        if timelist[1] == "11":
            season = "4"
            timeeleven+=1
        if timelist[1] == "12":
            season = "4"
            timetwelve+=1
        sql = "update datong_ugc set season = %s where id = %s"
        cursor.execute(sql, (season, id))
        conn.commit()

    conn.close()
    print(timeone,timetwo,timethree,timefour,timefive,timesix,timeseven,timeeight,timenine,timeten,timeeleven,timetwelve)
