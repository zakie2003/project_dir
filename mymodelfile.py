import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from flask import render_template,redirect,request,Flask

class MyModel:
    
    #Preprocessing the Data
    
    def __init__(self):

        f1 = pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv")
  
        f2 = pd.read_csv("IPL_Matches_Result_2008_2022.csv")
        
        
        f1 = f1[f1["overs"]<6]


        f1 = f1[f1["ID"]!=501265]
        f1 = f1[f1["ID"]!=829763]


        f2=f2[f2["ID"]!=501265]
        f2=f2[f2["ID"]!=829763]


        venue=f2.groupby("ID")["Venue"].unique().dropna()

        venue=pd.DataFrame(venue)


        f1_1=f1[f1["innings"]==1]
        f1_2=f1[f1["innings"]==2]

        run_1=f1_1.groupby("ID")["total_run"].sum().dropna()
        run_2=f1_2.groupby("ID")["total_run"].sum().dropna()

        in_1=f1_1.groupby("ID")["innings"].unique().dropna()
        in_2=f1_2.groupby("ID")["innings"].unique().dropna()


        bat_team_1=f1_1.groupby("ID")["BattingTeam"].unique().dropna()
        bat_team_2=f1_2.groupby("ID")["BattingTeam"].unique().dropna()

        striker_1=f1_1.groupby("ID")["batter"].unique().dropna()
        striker_2=f1_2.groupby("ID")["batter"].unique().dropna()

        bowler_1=f1_1.groupby("ID")["bowler"].unique().dropna()
        bowler_2=f1_2.groupby("ID")["bowler"].unique().dropna()
        
        
        va1=pd.concat([venue,in_1,bat_team_1,striker_1,bowler_1,run_1],axis=1)
        va2=pd.concat([venue,in_2,bat_team_2,striker_2,bowler_2,run_2],axis=1)


        fin_va=pd.concat([va1,va2])
        bat=fin_va["BattingTeam"].tolist()

        a=list(f2.groupby("ID")["Team1"].unique())
        b=list(f2.groupby("ID")["Team2"].unique())
        
        
        lst=[]
        for i in range(len(a)):
            l=[]
            l.append((a[i][0]))
            l.append((b[i][0]))
            lst.append(l)
        lst=lst+lst


        bowl=[]
        for i in range(len(bat)):
            p=[]
            for j in lst[i]:
                if(j!=bat[i]):
                    p.append(j)
            bowl.append(p)
        
        
        fin_va["venue"] = fin_va.Venue.apply(lambda x: ', '.join([str(i) for i in x]))
        fin_va["BattingTeam"]=fin_va.BattingTeam.apply(lambda x: ', '.join([str(i) for i in x]))
        fin_va["batter"]=fin_va.batter.apply(lambda x: ', '.join([str(i) for i in x]))
        fin_va["innings"]=fin_va.innings.apply(lambda x: ', '.join([str(i) for i in x]))
        fin_va["bowler"]=fin_va.bowler.apply(lambda x: ', '.join([str(i) for i in x]))
        
        bowl=pd.DataFrame(bowl,columns=["bowling_team"])
        fin_va=fin_va.reset_index(["ID"])

        refined=pd.concat([fin_va["venue"],fin_va["innings"],fin_va["BattingTeam"],bowl,fin_va["batter"],fin_va["bowler"],fin_va["total_run"]],axis=1)
        refined=refined.rename(columns={"Venue":"venue","innings":"innings","BattingTeam":"batting_team","bowling_team":"bowling_team","batter":"batsmen","bowler":"bowlers"})

        refined.to_csv("preprocess.csv")


    # Fitting The data      
    def fit(self,a):

        refined=pd.read_csv("preprocess.csv")
        p=refined
        
        ven=pd.get_dummies(p["venue"])
        bat_team=pd.get_dummies(p["batting_team"])
        bowl_team=pd.get_dummies(p["bowling_team"])
        
        
        batter=[]
        bowler=[]
        for i in p["batsmen"]:
            j=i.split(",")
            batter.append(len(j))
        for i in p["bowlers"]:
            j=i.split(",")
            bowler.append(len(j))

        dc={
            "batsmen":batter,
            "bowlers":bowler
        }
        dc=pd.DataFrame(dc)


        mix=pd.concat([ven,bat_team,bowl_team,dc,refined["total_run"]],axis=1)

        return mix
    # Prediction of the data
    
    def predict(self,inpf):

        refined=pd.read_csv("preprocess.csv",index_col=False)
        
        p=pd.concat([inpf,refined])
        p=p.drop(["Unnamed: 0"],axis=1)

        ven=pd.get_dummies(p["venue"])
        
        
        bat_team=pd.get_dummies(p["batting_team"])
        bowl_team=pd.get_dummies(p["bowling_team"])
        
        
        batter=[]
        bowler=[]
        
        for i in p["batsmen"]:
            if(isinstance(i[0],int)):
                batter.append(i[0])
            else:
                j=i.split(",")
                batter.append(len(j)-2)
        for i in p["bowlers"]:
            if(isinstance(i[0],int)):
                
                bowler.append(i[0])                           #Bug
            else:
                j=i.split(",")
                bowler.append(len(j))

        dc={
            "batsmen":batter,
            "bowlers":bowler
        }
        dc=pd.DataFrame(dc)
        dc.reset_index(drop=True,inplace=True)

        mix=pd.concat([ven,bat_team,bowl_team,refined["total_run"]],axis=1)
        mix.reset_index(drop=True,inplace=True)
        mix=pd.concat([mix,dc],axis=1)


        
        inp1=mix.iloc[0]

        

        mix=mix.tail(-1)
        
        
        run=mix["total_run"]
        
        
        mix=mix.drop(["total_run"],axis=1)
        inp1=inp1.drop(["total_run"])
        inp1["batsmen"]=int(inpf["batsmen"])-2
        inp1["bowlers"]=int(inpf["bowlers"])
        
        # MODEL SELECTION

        x_train,x_test,y_train,y_test=train_test_split(mix.values,run.values,test_size=0.01)
        md=GradientBoostingRegressor()
        md.fit(x_train,y_train)
        
        result=[]
        joblib.dump(md,"model.sav")
        load_md=joblib.load("model.sav")
        r1=load_md.predict([inp1])

        result.append(r1)
        return(result)
    

app=Flask(__name__)
@app.route("/",methods=('GET', 'POST'))

def inp():   
    if request.method=="POST":
        inpf={"venue":request.form["venue"],"innings":request.form["inning"],"batting_team":request.form["bat"],"bowling_team":request.form["bowl"],"batsmen":request.form["wic"],"bowlers":request.form["numbboler"]}
        inpf=pd.DataFrame.from_dict([inpf])

        preds=pd.read_csv("test_file.csv")

        aaa=MyModel()
        zzz=aaa.predict(inpf)
        
        return render_template("index.html",result=f'The Score of the input is {int(zzz[0])}')  

    return render_template("index.html")     
           
        
app.run(host='0.0.0.0',debug=True)       
