import pandas as pd
import numpy as np
#from sklearn import datasets
#import matplotlib.pyplot as plt
import MysqlDf

def SetLabels():
        Labels=[-10,-9,-8,-7,-6,-5,-4,-3,-2.7,-2.4,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3,4,5,6,7,8,9,10]
        #Labels=[ -2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]
        #Set makes lables as string  --- Important to note
        LabelDict={label:idx for idx, label in enumerate ((Labels))}
        reverse_dictionary = {idx: label for idx, label in enumerate((Labels))}
        #ZeroVec = [0 for _ in range(len(Labels))]
        #print(LabelDict)
        return LabelDict, reverse_dictionary,len(Labels)


def GetOneHot(PctChg,LabelDict,LibrarySize):
        if(PctChg<-10.0):
                label=-10
        elif(PctChg>10.0):
                label=10
        #else:
        elif((PctChg<-3) or (PctChg>3)):
                label=round(PctChg)
        else:
                label=round(0.3*int(PctChg/0.3),1)
                #label = round(0.5 * int(PctChg / 0.5), 1)

        idx=LabelDict[label]
        OneHotVec=[0 for _ in range(LibrarySize)]
        OneHotVec[idx]=1
        #print(OneHotVec)
        #print(label, idx)

        return OneHotVec




def Parse_intraday():
        """
        #data = "C:/tmp/intraday"
        data = "C:/tmp/eqintra.txt"
        df_intra = pd.read_csv(data, sep="\t",
                     names=["SYMBOL","OPEN","HIGH","LOW","CLOSE","VOLUME","EMA20","EMA50","EMA100","EMA20_SLOP1","EMA50_SLOP1","CUTOVER1",
                            "EMA20_SLOP2","EMA50_SLOP2","CUTOVER2","EMA20_SLOP3","EMA50_SLOP3","CUTOVER3",
                             "EMA20_30M","EMA50_30M","EMA100_30M","EMA20_SLOP1_30M","EMA50_SLOP1_30M","CUTOVER1_30M",
                            "EMA20_SLOP2_30M", "EMA50_SLOP2_30M", "CUTOVER2_30M","EMA20_SLOP3_30M","EMA50_SLOP3_30M","CUTOVER3_30M",
                            "TIMESTAMP"], header=1)
                            """
        df_intra=MysqlDf.GetMysqlDF("INTRADAY","select * from EQ_INTRASTATS")

        df_intra['TIMESTAMP'] = pd.to_datetime(df_intra['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
        df_intra['TIME'] = df_intra["TIMESTAMP"].apply(lambda x: x.time().strftime('%H:%M:%S'))
        df_intra = df_intra[(df_intra["TIME"]=="10:00:00")]
        df_intra = df_intra [["SYMBOL","CLOSE","EMA20","EMA50","EMA20_SLOP1","EMA50_SLOP1","CUTOVER1",
                            "EMA20_SLOP2","EMA50_SLOP2","CUTOVER2","EMA20_30M","EMA50_30M","EMA20_SLOP1_30M","EMA50_SLOP1_30M",
                              "CUTOVER1_30M","EMA20_SLOP2_30M", "EMA50_SLOP2_30M", "CUTOVER2_30M","TIMESTAMP"]]
        #print(df_intra[(df_intra["SYMBOL"]=="GMRINFRA")])

        df_intra = df_intra.sort_values(["SYMBOL", "TIMESTAMP"], ascending=[True, True])
        df_intra["TIMESTAMP"] = pd.to_datetime(df_intra["TIMESTAMP"].apply(lambda x: x.date().strftime("%Y-%m-%d")),format="%Y-%m-%d")
        #df_intra = df_intra[(df_intra["TIMESTAMP"] > '2018-01-11')]

        #print(type(df_intra["TIMESTAMP"]))
        """
        df_intra = df_intra[["SYMBOL","EMA20","EMA50","EMA100","EMA20_SLOP1","EMA20_SLOP2",
                            "EMA20_30M","EMA50_30M","EMA100_30M","EMA20_SLOP1_30M","EMA20_SLOP2_30M","TIMESTAMP"]]
        """


        return df_intra

def Parse_data(Train):
        LableDict, RevDict, ZeroVec= SetLabels()
        df_intra = Parse_intraday()
        df_intra = df_intra [["SYMBOL","TIMESTAMP","EMA20","EMA50","EMA20_SLOP1","EMA50_SLOP1","EMA20_30M","EMA50_30M",
                              "CUTOVER1","EMA20_SLOP2","EMA50_SLOP2","CUTOVER2","EMA20_SLOP2_30M","EMA20_SLOP1_30M",
                              "EMA50_SLOP1_30M","EMA50_SLOP2_30M"]]
        #print(df_intra)

        print(df_intra.shape)
        df_in = Parse_Bhav()
        #print(df_in)
        df_trend=Parse_Trend()
        print(df_in.shape)
        print(df_trend.shape)
        df_in = df_in[["SYMBOL", "TIMESTAMP", "CLOSE","AVGP","OPEN","HIGH","LOW","MINPRICE","MAXPRICE", "EMA_12", "EMA_26", "EMA12_SLOP1",
                       "EMA26_SLOP1", "CUTOVER1", "EMA12_SLOP2", "EMA26_SLOP2", "CUTOVER2","SSTO_D","SSTO_K","MGAP"]]

        df_final = df_in[(df_in["TIMESTAMP"] > '2017-10-01') ]
        df_final = df_in[(df_in["SYMBOL"] == 'NIFTY')]
        df_final = pd.merge(df_final, df_intra, how="inner", on=["SYMBOL", "TIMESTAMP"])
        print(df_final.shape)

        df_final = pd.merge(df_final,df_trend,  how="inner", right_on=["SYMBOL", "PROCDT"],
                            left_on=["SYMBOL", "TIMESTAMP"])
        print("Before printing final set")
        print(df_final)
        #print(df_final[df_final.PROCDT==NaN])
        df_final.to_csv(r'c:\tmp\pandas.txt', header=None, index=None, sep=' ', mode='a')
        df_final = df_final.sort_values(['SYMBOL', 'TIMESTAMP'], ascending=[True, True]).reset_index()

        df_final = df_final.iloc[-20:,:]
        print(df_final)
        """
        df_plot = df_final[(df_final["SYMBOL"] == "ITC")]
        df_plot = df_plot[["TIMESTAMP", "EMA20", "EMA50", "EMA20_30M",
                           "EMA50_30M", "EMA_12","EMA_26"]]
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA20"], label="EMA20")
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA50"], label="EMA50")
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA20_30M"], label="EMA20_SLOP1")
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA50_30M"], label="EMA50_SLOP1")
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA_12"], label="EMA_12")
        plt.plot(df_plot["TIMESTAMP"], df_plot["EMA_26"], label="EMA_26")
        plt.legend(["EMA20", "EMA50","EMA20_30M","EMA50_30M","EMA_12","EMA_26"], loc=4)
        # df_plot.plot(style=['o', 'rx'])
        plt.show()
        """
        df_final.loc[pd.isnull(df_final["SLOP_1_UP"]), "SLOP_1_UP"] = df_final["SLOP_1_DN"]
        df_final.loc[pd.isnull(df_final["SUPPORT1"]), "SUPPORT1"] = (df_final["SLOP_1_DN"] + df_final["MINPRICE"])
        df_final.loc[pd.isnull(df_final["SLOP_1_DN"]), "SLOP_1_DN"] = df_final["SLOP_1_UP"]
        df_final.loc[pd.isnull(df_final["RESISTANCE1"]), "RESISTANCE1"] = (df_final["SLOP_1_UP"] + df_final["MAXPRICE"])
        df_final.loc[pd.isnull(df_final["SLOP_2_UP"]), "SLOP_2_UP"] = df_final["SLOP_2_DN"]
        df_final.loc[pd.isnull(df_final["SUPPORT2"]), "SUPPORT2"] = (df_final["SLOP_2_DN"] + df_final["MINPRICE"])
        df_final.loc[pd.isnull(df_final["SLOP_2_DN"]), "SLOP_2_DN"] = df_final["SLOP_2_UP"]
        df_final.loc[pd.isnull(df_final["RESISTANCE2"]), "RESISTANCE2"] = (df_final["SLOP_2_UP"] + df_final["MAXPRICE"])
        df_final["LONG_SLOP1"] = df_final["EMA12_SLOP1"] - df_final["EMA26_SLOP1"]
        df_final["LONG_SLOP2"] = df_final["EMA12_SLOP2"] - df_final["EMA26_SLOP2"]
        df_final["MID_SLOP1"] = df_final["EMA20_SLOP1_30M"] - df_final["EMA50_SLOP1_30M"]
        df_final["MID_SLOP2"] = df_final["EMA20_SLOP2_30M"] - df_final["EMA50_SLOP2_30M"]
        df_final["SHORT_SLOP1"] = df_final["EMA20_SLOP1"] - df_final["EMA50_SLOP1"]
        df_final["SHORT_SLOP2"] = df_final["EMA20_SLOP2"] - df_final["EMA50_SLOP2"]

        df_final = df_final.fillna(0)
        df_final = df_final.replace(0, 0.0001)


        #print(df_final)
        symbols=df_final.SYMBOL.unique()
        features = []

        output = []
        outputclass=[]
        Data_final= None
        j=0
        for symbol in  symbols:

                Data=df_final[(df_final["SYMBOL"] == symbol)]


                #print(Data)
                Data_ratio = Data[[key for key in dict(Data.dtypes) if dict(Data.dtypes)[key] in ['float64', 'int64']]].pct_change()
                Data_ratio["SYMBOL"]=Data["SYMBOL"]
                Data_ratio["TIMESTAMP"] =Data["TIMESTAMP"]
                Data_ratio["CLV"]  = Data["CLOSE"]
                Data["CLV"] = Data["EMA12_SLOP1"] - Data["EMA26_SLOP1"]

                Data_ratio.loc[Data["SLOP_1_UP"].diff() > 0, "R1_BROKEN"] = 1
                Data_ratio.loc[Data["SLOP_2_UP"].diff() > 0, "R2_BROKEN"] = 1
                Data_ratio.loc[Data["SLOP_1_DN"].diff() > 0, "S1_BROKEN"] = 1
                Data_ratio.loc[Data["SLOP_2_DN"].diff() > 0, "S2_BROKEN"] = 1
                df =Data [["EMA20", "EMA50","EMA20_30M","EMA50_30M","EMA_12","EMA_26","SLOP_1_UP","SLOP_1_DN","SLOP_2_UP","SLOP_2_DN",
                           "LONG_SLOP1","MID_SLOP1","SHORT_SLOP1","LONG_SLOP2","MID_SLOP2","SHORT_SLOP2"]]
                #normalized_df = (df - df.mean()) / df.std()
                normalized_df = (df - df.min()) / (df.max()-df.min())
                #print(normalized_df)
                #Data_ratio.loc[Data["SLOP_2_DN"].diff() > 0, "S1_BROKEN_F"] = 1
                Data_ratio.loc[Data["TOT_1_UP"] > Data["TOT_1_UP"].mean(), "S1_BROKEN_F"] = 1
                Data_ratio.loc[Data["TOT_1_DN"] > Data["TOT_1_DN"].mean(), "R1_BROKEN_F"] = 1

                Data["BETA1"] = Data_ratio["CLOSE"].std()
                Data["BETA2"] = Data_ratio["CLOSE"].mean()

                #Data_ratio["TOT_UP"]=(Data["TOT_1_UP"] - Data["TOT_1_UP"].mean())/Data["TOT_1_UP"].std()
                #Data_ratio["TOT_DN"] = (Data["TOT_1_DN"] - Data["TOT_1_DN"].mean()) / Data["TOT_1_DN"].std()
                Data["PCTCHG"] =round(Data_ratio["CLOSE"]*100,3)

                #print(Data_ratio)
                Data_ratio = Data_ratio.fillna(0)
                Data_ratio = Data_ratio.replace(0, 0.0001)

                #F1 = np.array(Data["OPEN"])
                #F2 =np.array(Data["CLOSE"])
                #F3 = np.array(Data["MINPRICE"])
                #F4 = np.array(Data["MAXPRICE"])
                #F5 = np.array(Data["AVGP"])
                #F6 = np.array(Data["RESISTANCE1"])
                #F7 = np.array(Data["SUPPORT1"])
                F1 =  np.array((Data_ratio["R1_BROKEN"]))
                F2 = np.array((Data_ratio["R2_BROKEN"]))
                F3 = np.array((Data_ratio["S1_BROKEN"]))
                F4 = np.array((Data_ratio["S2_BROKEN"]))
                F5 = np.array(Data_ratio["SLOP_1_UP"])
                F6 = np.array(Data_ratio["SLOP_1_DN"])
                F5 = np.array(Data_ratio["SLOP_2_UP"])
                F6 = np.array(Data_ratio["SLOP_2_DN"])
                F7 = np.array(normalized_df[ "SHORT_SLOP1"])
                F8 = np.array(normalized_df["SHORT_SLOP2"])
                F9 = np.array(normalized_df["MID_SLOP1"])
                F7 = np.array(normalized_df["EMA20"])
                F8 = np.array(normalized_df["EMA50"])
                #F9 = np.array(normalized_df["EMA20_30M"])
                #F7 = np.array(Data_ratio["R1_BROKEN_F"])
                #F8 = np.array(Data_ratio["S1_BROKEN_F"])
                F10 = np.array(Data["RESISTANCE1"])
                F11 = np.array(Data["SUPPORT1"])
                F12 = np.array(Data["RESISTANCE2"])
                F13 = np.array(Data["SUPPORT2"])
                F14= np.array(normalized_df["MID_SLOP2"])
                F15 = np.array(normalized_df["LONG_SLOP1"])
                F16 = np.array(normalized_df["LONG_SLOP2"])
                #F14 = np.array(normalized_df["EMA50_30M"])
                #F15 = np.array(normalized_df["EMA_12"])
                #F16 = np.array(normalized_df["EMA_26"])
                F10 = np.array(Data["SSTO_K"])
                #F8 = np.array(Data_ratio["SSTO_D"])
                F9 = np.array(Data_ratio["SLOP_1_UP"])
                F14 = np.array(Data_ratio["SLOP_1_DN"])
                F15 = np.array(Data_ratio["SLOP_2_UP"])
                F16 = np.array(Data_ratio["SLOP_2_DN"])

                F9 = np.array(Data_ratio["LONG_SLOP1"])
                F14 = np.array(Data_ratio["LONG_SLOP2"])
                F15 = np.array(Data_ratio["MID_SLOP1"])
                F16 = np.array(Data_ratio["MID_SLOP2"])
                FC = np.array(Data_ratio["SHORT_SLOP1"])
                FD = np.array(Data_ratio["SHORT_SLOP2"])
                #F9 = np.array(Data_ratio["EMA20"])
                #F14 = np.array(Data_ratio["EMA50"])
                #F15 = np.array(Data_ratio["EMA20_30M"])
                #F16 = np.array(Data_ratio["EMA50_30M"])
                #FC = np.array(Data_ratio["EMA_12"])
                #FD = np.array(Data_ratio["EMA_26"])



                #F8 = np.array((Data["EMA20_30M"]))
                #F9 = np.array(Data["EMA50_30M"])
                #F7 = np.array(Data["SSTO_D"])
                #F5 = np.array(Data_ratio["SSTO_D"])
                #F14 = np.array(Data["OPEN"])
                F17 = np.array(Data["CLOSE"])
                F18 = np.array(Data["BETA1"])
                F19 = np.array(Data["BETA2"])

                #F8 = np.array(Data_ratio["SLOP_2_UP"])
                #F9 = np.array(Data_ratio["SLOP_2_DN"])
                #F6 = np.array(Data_ratio["EMA20_SLOP2_30M"])
                #F7 = np.array(Data_ratio["EMA20_SLOP1_30M"])
                #F8 = np.array(Data["SSTO_D"])
                #F6 = np.array(Data["HIGH"])
                #F7 = np.array(Data["LOW"])
                #F8 = np.array(Data["CLOSE"])

                F11 = np.array(Data_ratio["EMA12_SLOP1"])
                F12 = np.array(Data_ratio["EMA12_SLOP2"])
                F13 = np.array(Data_ratio["EMA26_SLOP1"])
                F14 = np.array(Data_ratio["EMA26_SLOP2"])
                F15 = np.array(Data_ratio["EMA50_SLOP1_30M"])
                F16 = np.array(Data_ratio["EMA50_SLOP2_30M"])
                F17 = np.array(Data_ratio["EMA20_SLOP1_30M"])
                F18 = np.array(Data_ratio["EMA20_SLOP2_30M"])
                F19 = np.array(Data_ratio["EMA20_SLOP1"])
                F20 = np.array(Data_ratio["EMA20_SLOP2"])
                F21 = np.array(Data_ratio["EMA50_SLOP1"])
                F22 = np.array(Data_ratio["EMA50_SLOP2"])


                yData=np.array(Data_ratio["CLOSE"])
                #xData = np.array((Data["Macd"]))

                featuresize = len(Data["CLOSE"]) - (len(Data["CLOSE"])) % 20
               # Data_final =  pd.concat([Data_final, Data_ratio.iloc[5:,:]])
                Data_final = pd.concat([Data_final, Data.iloc[0:featuresize, :]])
                if Train:
                      for i in range(0, featuresize-21):
                             #tmpfeatures=[]
                             #tmpoutput=[]
                             #tmpoutputclass = []
                             inset = []
                             outset = []
                             outclassset = []

                             for j in range (0,20):
                                  k=i+j
                                  #fv = [F9[k], F14[k],F15[k], F16[k] ,FC[k],FD[k] ]
                                  fv = [F11[k], F12[k],F13[k], F14[k] ,F15[k],F16[k],F17[k],F18[k],F19[k],F20[k],F21[k],F22[k] ]
                                  inset.append(fv)
                                  outset.append([yData[k + 1]])
                                  OneHot = GetOneHot(yData[k + 1]*100, LableDict, ZeroVec)
                                  outclassset.append(OneHot)



                             features.append(inset)
                             output.append(outset)
                             outputclass.append(outclassset)
                             """
                             if Train:
                                     output.append([yData[i + 1]])
                                     #output.append([F1[i+5]])
                                     OneHot = GetOneHot(yData[i + 1]*100, LableDict, ZeroVec)
                             else:

                                     if(i==featuresize-1) :
                                          output.append(0)
                                     else:
                                          output.append([yData[i + 1]])
                                     #output.append([F1[i+5]])
                                          OneHot = GetOneHot(yData[i + 1]*100, LableDict, ZeroVec)

                             outputclass.append(OneHot)
                             """
                else:
                      for k in range(0,20) :
                             fv = [F9[k], F14[k],F15[k], F16[k] ,FC[k],FD[k] ]
                             fv = [F11[k], F12[k],F13[k], F14[k] ,F15[k],F16[k],F17[k],F18[k],F19[k],F20[k],F21[k],F22[k] ]
                             features.append(fv)
                             output.append([yData[k]])
                             OneHot = GetOneHot(yData[k]*100, LableDict, ZeroVec)
                             outputclass.append(OneHot)


                #features.append(tmpfeatures[(len(tmpfeatures)-len(tmpfeatures)%20)])
                #output.append(tmpoutput[(len(tmpoutput) - len(tmpoutput) % 20)])
                #outputclass.append(tmpoutputclass[(len(tmpoutputclass) - len(tmpoutputclass) % 20)])

                Data_final = Data_final[["SYMBOL","TIMESTAMP","CLOSE","OPEN","PCTCHG"]]
                                         #,"BETA2","SUPPORT1","RESISTANCE1","SLOP_1_UP","SLOP_1_DN","CUTOVER1","CUTOVER2","EMA12_SLOP1","EMA26_SLOP1",
                                         #"SDATE_1_UP", "EDATE_1_UP", "SPRICE_1_UP","EPRICE_1_UP","SDATE_1_DN", "EDATE_1_DN", "SPRICE_1_DN","EPRICE_1_DN"  ]]
        return ( RevDict, np.array(features), np.array(output), np.array(outputclass),Data_final.reset_index())


def Parse_Fno():
        """
        #data = "C:/tmp/fodata"
        data = "C:/tmp/fnodata"
        df_fno = pd.read_csv(data, sep="\t", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
                               names=["EXPIRY_DT", "SYMBOL", "OPTION_TYP", "STRIKE_PR", "OPEN",
                                      "CLOSE", "AVG_PR", "INV_AMT", "OPEN_INT", "CHG_IN_OI", "TIMESTAMP"], header=1)
                                      """
        df_fno=MysqlDf.GetMysqlDF("FNO","select * from FO_STATSN")

        df_fno['TIMESTAMP'] = pd.to_datetime(df_fno['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
        df_fno['EXPIRY_DT'] = pd.to_datetime(df_fno['EXPIRY_DT'], format='%d-%b-%Y')
        df_fno = df_fno.sort_values(["SYMBOL","TIMESTAMP","OPTION_TYP","STRIKE_PR","EXPIRY_DT"],
                                    ascending=[True, True, True, True, True])
        df_fu = df_fno[(df_fno["OPTION_TYP"]=="XX")]

        df_fu = df_fu.groupby(["SYMBOL","TIMESTAMP", "OPTION_TYP", "STRIKE_PR", "EXPIRY_DT"])["INV_AMT"].sum().reset_index()
        df_fu.loc[:, "CUMSUM"] = df_fu.groupby(["SYMBOL", "OPTION_TYP","STRIKE_PR","EXPIRY_DT"])["INV_AMT"].cumsum()
        df_fu.loc[df_fu["EXPIRY_DT"] == df_fu["TIMESTAMP"], "OFFSET"] = df_fu["CUMSUM"]
        df_fu = df_fu.fillna(0)
        #print(df_fu[(df_fu["SYMBOL"] == "NHPC")])
        #df_fu.loc[:, "CUMOFFSET"] = df_fu.groupby(["SYMBOL", "OPTION_TYP", "STRIKE_PR","EXPIRY_DT"])["OFFSET"].cumsum()
        #print(df_fu.sort_values(['SYMBOL', 'TIMESTAMP'], ascending=[True, True]))
        df_fu_agg = df_fu.groupby(["SYMBOL", "TIMESTAMP"]).sum().reset_index()
        #df_fu_agg = df_fu_agg[(df_fu_agg["SYMBOL"]=="ZEEL")]
        #print(df_fu_agg[["SYMBOL","TIMESTAMP","INV_AMT","CUMSUM","OFFSET"]])

        df_op_ce = df_fno[(df_fno["OPTION_TYP"]=="CE")]
        #print(df_op_ce)
        df_op_ce = df_op_ce.groupby(["SYMBOL","TIMESTAMP","OPTION_TYP", "STRIKE_PR","EXPIRY_DT"])["INV_AMT"].sum().reset_index()
        df_op_ce.loc[ :,"CUMSUM"] = df_op_ce.groupby(["SYMBOL","OPTION_TYP", "STRIKE_PR","EXPIRY_DT"])["INV_AMT"].cumsum()
        df_op_ce.loc[df_op_ce["EXPIRY_DT"] == df_op_ce["TIMESTAMP"], "OFFSET"] = df_op_ce["CUMSUM"]

        #df_op_ce.loc[:, "CUMOFFSET"] = df_op_ce.groupby(["SYMBOL", "OPTION_TYP", "STRIKE_PR"])["OFFSET"].cumsum()
        #print(df_op_ce[(df_op_ce["SYMBOL"] == "ACC") & (df_op_ce["TIMESTAMP"] < "2017-05-20")])
        #print(df_op_ce[(df_op_ce["SYMBOL"]=="ACC")])
        df_op_ce=df_op_ce[["SYMBOL","TIMESTAMP","EXPIRY_DT","INV_AMT","STRIKE_PR","CUMSUM","OFFSET"]]
        #print(df_op_ce)


        df_op_pe = df_fno [ (df_fno["OPTION_TYP"]=="PE")]
        df_op_pe = df_op_pe.groupby(["SYMBOL", "TIMESTAMP", "OPTION_TYP", "STRIKE_PR", "EXPIRY_DT"])["INV_AMT"].sum().reset_index()
        df_op_pe.loc[:,"CUMSUM"] = df_op_pe.groupby(["SYMBOL", "OPTION_TYP", "STRIKE_PR", "EXPIRY_DT"])["INV_AMT"].cumsum()
        df_op_pe.loc[df_op_pe["EXPIRY_DT"]==df_op_pe["TIMESTAMP"], "OFFSET"] = df_op_pe["CUMSUM"]

        #df_op_pe.loc[:,"CUMOFFSET"]=df_op_pe.groupby(["SYMBOL", "OPTION_TYP", "STRIKE_PR"])["OFFSET"].cumsum()
        #print(df_op_pe[(df_op_pe["SYMBOL"]=="ACC") & (df_op_pe["TIMESTAMP"]<"2017-05-20")])
        df_op_pe = df_op_pe[["SYMBOL", "TIMESTAMP", "EXPIRY_DT", "INV_AMT", "STRIKE_PR", "CUMSUM", "OFFSET"]]


        df_op = pd.merge(df_op_ce, df_op_pe, how="outer", on= ["SYMBOL","TIMESTAMP","EXPIRY_DT","STRIKE_PR"], suffixes=("_CE", "_PE")).fillna(0)
        df_op = df_op[["SYMBOL", "TIMESTAMP", "STRIKE_PR", "EXPIRY_DT", "CUMSUM_CE","CUMSUM_PE","OFFSET_CE","OFFSET_PE"]]
        df_op = df_op.fillna(0.001)
        #print(df_op[(df_op["SYMBOL"] =="ZEEL")])
        #df_fno = pd.merge(df_op, df_fu, how="outer", on=["SYMBOL", "TIMESTAMP"],suffixes=("_OP", "_FU")).fillna(0)
        df_fu = df_fu[(df_fu["TIMESTAMP"] > '2017-08-01')]
        df_op = df_op[(df_op["TIMESTAMP"] > '2017-08-01')]
        #print(df_op[(df_op["SYMBOL"] == "ZEEL")])

        return (df_fu_agg, df_op)



def Parse_Trend():
        """#data ="C:/tmp/data1"
        #data = "C:/tmp/trendata"
        data = "C:/tmp/eqtrend.txt"
        df_trend = pd.read_csv(data, sep="\t", usecols=[0,1,2,3,4,5,6,7,8,9],
                     names=["SYMBOL", "SLOP_TYPE", "LINE_NO", "SDATE", "EDATE", "SPRICE", "EPRICE", "TOT", "SLOP", "PROCDT"], header=1)
                     """
        df_trend=MysqlDf.GetMysqlDF("TREND","select * from EQ_TRENDN")

        #print(df_trend)
        #df_trend['EDate'] = pd.to_datetime(df_trend['EDate'], format='%Y-%m-%d')
        #df_trend['ProcDt']=pd.to_datetime(df_trend['ProcDt'],format='%Y-%m-%d')
        df_up_1 = df_trend[ (df_trend.SLOP_TYPE =="UP") & (df_trend.LINE_NO==1) ]
        #print(df_up_1)
        df_up_1.loc[:,"SUPPORT1"] = df_up_1["SPRICE"] + df_up_1["SLOP"]
        df_up_1 = df_up_1.replace(0,0.0001)
        df_up_2 = df_trend[(df_trend.SLOP_TYPE == "UP") & (df_trend.LINE_NO == 2)]
        #print(df_up_2)
        df_up_2 = df_up_2.replace(0, 0.0001)
        df_up_merge = pd.merge(df_up_1,df_up_2, how="left",on=["SYMBOL","PROCDT"], suffixes=("_1", "_2"))
        df_up_merge.loc[pd.isnull(df_up_merge["SLOP_2"]), "SLOP_2"] = df_up_merge["SLOP_1"]
        df_up_merge.loc[:,"SUPPORT2"] = df_up_merge["SPRICE_2"]+(df_up_merge["TOT_1"]+df_up_merge["TOT_2"]+1)*df_up_merge["SLOP_2"]
        #print(df_up_merge)

        df_dn_1 = df_trend[(df_trend["SLOP_TYPE"]=="DN") & (df_trend["LINE_NO"] == 1)]
        #df_dn_1=df_dn_1.replace(0,0.0001)
        #print(df_dn_1)
        df_dn_1.loc[:,"RESISTANCE1"] = df_dn_1["SPRICE"] + df_dn_1["SLOP"]
        df_dn_2 = df_trend[(df_trend["SLOP_TYPE"] == "DN") & (df_trend["LINE_NO"] == 2)]
        #df_up_2 = df_up_2.replace(0, 0.0001)
        #print(df_dn_2)
        df_dn_merge = pd.merge(df_dn_1, df_dn_2, how="left", on=["SYMBOL", "PROCDT"], suffixes=("_1", "_2"))
        df_dn_merge.loc[pd.isnull(df_dn_merge["SLOP_2"]), "SLOP_2"] = df_dn_merge["SLOP_1"]
        df_dn_merge.loc[:, "RESISTANCE2"] = df_dn_merge["SPRICE_2"] + (df_dn_merge["TOT_1"] + df_dn_merge["TOT_2"] + 1) *df_dn_merge["SLOP_2"]
        #print(df_dn_merge)
        df_merge = pd.merge(df_up_merge,df_dn_merge,how="outer",on=["SYMBOL","PROCDT"],suffixes=("_UP", "_DN"))

        #print(df_merge)
        #print(df_merge[(df_merge.SLOP_x==0)])
        #df_merge=df_merge.replace(0, 10000)
        df_merge["PROCDT"]=pd.to_datetime(df_merge["PROCDT"],format='%Y-%m-%d')
        df_merge.sort_values(['SYMBOL', 'PROCDT'], ascending=[True, True])
        #print(df_merge[["PROCDT","SLOP_x","SLOP_y"]])
        return df_merge


def Parse_Bhav():
        """data = "C:/tmp/eqdata.txt"
        #data = "C:/tmp/eqdata"
        df_bh = pd.read_csv(data, sep="\t", names =["SYMBOL","OPEN","HIGH","LOW","CLOSE","MINPRICE","MAXPRICE","BTSTDIFF",
                                                    "AVGP","EMA_26","EMA_12","MACD","EMA_9","SSTO_K","SSTO_D","EMA12_SLOP1",
                                                    "EMA26_SLOP1","CUTOVER1","EMA12_SLOP2","EMA26_SLOP2","CUTOVER2","EMA12_SLOP3",
                                                    "EMA26_SLOP3","CUTOVER3","TIMESTAMP"],  header=1)
                                                    """
        df_bh=MysqlDf.GetMysqlDF("EQ_BHAV","select * from EQ_STATSN")

        df_bh['TIMESTAMP'] = pd.to_datetime(df_bh['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')

        df_bh["MGAP"]=df_bh["MACD"]-df_bh["EMA_9"]
        df_bh = df_bh[(df_bh["TIMESTAMP"] >= "2017-06-01")]
        df_bh = df_bh.replace(0, 0.0001)
        #print(df_bh)
        df_bh.sort_values(['SYMBOL', 'TIMESTAMP'], ascending=[True, True])
        #print(df_bh)
        return df_bh


def Parse_mum(Train):
        LableDict, RivDict, ZeroVec = SetLabels()
        df_trend = Parse_Trend()
        df_bhav = Parse_Bhav()
        (df_fu,df_op) = Parse_Fno()
        #df_intra = Parse_intraday()

        df_bhav_op = pd.merge(df_op, df_bhav, how="inner", on=["SYMBOL", "TIMESTAMP"])
        #print(df_bhav_op)
        df_bhav_op = df_bhav_op.sort_values(['SYMBOL', 'TIMESTAMP'], ascending=[True, True])


        df_bhav_op.loc[df_bhav_op["CLOSE"] <= df_bhav_op["STRIKE_PR"], "INV_ABOVE_CE"] = df_bhav_op["CUMSUM_CE"] #-df_bhav_op["CUMOFFSET_CE"]
        df_bhav_op.loc[df_bhav_op["CLOSE"] > df_bhav_op["STRIKE_PR"], "INV_BELOW_CE"] = df_bhav_op["CUMSUM_CE"] #-df_bhav_op["CUMOFFSET_CE"]
        df_bhav_op.loc[df_bhav_op["CLOSE"] <= df_bhav_op["STRIKE_PR"], "INV_ABOVE_PE"] = df_bhav_op["CUMSUM_PE"] #-df_bhav_op["CUMOFFSET_PE"]
        df_bhav_op.loc[df_bhav_op["CLOSE"] > df_bhav_op["STRIKE_PR"], "INV_BELOW_PE"] = df_bhav_op["CUMSUM_PE"] #-df_bhav_op["CUMOFFSET_PE"]
        #df_bhav_op=df_bhav_op.fillna(0)

        #df_bhav_op_agg =df_bhav_op[["SYMBOL","TIMESTAMP","OFFSET_PE","CUMSUM_PE","OFFSET_CE","CUMSUM_CE","INV_ABOVE_PE","INV_BELOW_PE"]]
        df_bhav_op_agg = df_bhav_op.groupby(["SYMBOL","TIMESTAMP"]).sum().reset_index()
        df_bhav_op_agg=df_bhav_op_agg.replace(0,0.001)

        df_bhav_op_agg.loc[:,"PUTCALL_RATIO"] = df_bhav_op_agg["CUMSUM_PE"]/df_bhav_op_agg["CUMSUM_CE"]
        df_bhav_op_agg.loc[:,"AB_RATIO_PE"]  =df_bhav_op_agg["INV_ABOVE_PE"]/df_bhav_op_agg["INV_BELOW_PE"]
        df_bhav_op_agg.loc[:, "AB_RATIO_CE"] = df_bhav_op_agg["INV_ABOVE_CE"] / df_bhav_op_agg["INV_BELOW_CE"]
        df_bhav_op_agg.loc[:,"AA_RATIO"] = df_bhav_op_agg["INV_ABOVE_CE"] / df_bhav_op_agg["INV_ABOVE_PE"]
        df_bhav_op_agg.loc[:,"BB_RATIO"] = df_bhav_op_agg["INV_BELOW_CE"] / df_bhav_op_agg["INV_BELOW_PE"]
        df_bhav_op_agg = df_bhav_op_agg[["SYMBOL","TIMESTAMP","PUTCALL_RATIO","AB_RATIO_PE","AB_RATIO_CE","AA_RATIO","BB_RATIO"]]
        #df_bhav_op_agg.fillna(0)
        df_bhav_op_agg_new = pd.merge(df_bhav_op_agg, df_bhav, how="inner", on=["SYMBOL", "TIMESTAMP"])
        print(df_bhav_op_agg_new)
        df_bhav_fno = pd.merge(df_bhav_op_agg_new, df_fu, how="inner", on=["SYMBOL", "TIMESTAMP"])


        #print(df_bhav_op_agg[(df_bhav_op_agg["SYMBOL"]=="ACC")])

        #print(df_bhav_op[["SYMBOL","TIMESTAMP","EXPIRY_DT", "STRIKE_PR","CLOSE","INV_ABOVE_PE","INV_BELOW_PE"]])
        #df_bhav_fno = pd.merge(df_fu, df_bhav_op, how="outer", on=["SYMBOL", "TIMESTAMP","EXPIRY_DT"]).fillna(0)
        #print(df_bhav_fno[["SYMBOL", "TIMESTAMP", "EXPIRY_DT",  "CLOSE", "INV_ABOVE_CE", "INV_BELOW_CE"]])
        #df_bhav_fno_intra = pd.merge(df_intra, df_bhav_fno, how="inner", on=["SYMBOL", "TIMESTAMP"])
        df_final = pd.merge(df_trend, df_bhav_fno, how="inner", left_on=["SYMBOL", "PROCDT"],
                            right_on=["SYMBOL", "TIMESTAMP"])
        df_final.loc[pd.isnull(df_final["SLOP_1_UP"]), "SLOP_1_UP"] = df_final["SLOP_1_DN"]
        df_final.loc[pd.isnull(df_final["SUPPORT1"]), "SUPPORT1"] = df_final["SLOP_1_DN"]+df_final["MINPRICE"]
        df_final.loc[pd.isnull(df_final["SLOP_1_DN"]), "SLOP_1_DN"] = df_final["SLOP_1_UP"]
        df_final.loc[pd.isnull(df_final["RESISTANCE1"]), "RESISTANCE2"] = df_final["SLOP_2_UP"]+df_final["MAXPRICE"]
        df_final.loc[pd.isnull(df_final["SLOP_2_UP"]), "SLOP_1_UP"] = df_final["SLOP_2_DN"]
        df_final.loc[pd.isnull(df_final["SUPPORT2"]), "SUPPORT2"] = df_final["SLOP_2_DN"] + df_final["MINPRICE"]
        df_final.loc[pd.isnull(df_final["SLOP_2_DN"]), "SLOP_2_DN"] = df_final["SLOP_2_UP"]
        df_final.loc[pd.isnull(df_final["RESISTANCE1"]), "RESISTANCE2"] = df_final["SLOP_2_UP"] + df_final["MAXPRICE"]


        #df_nan = df_trend[pd.isnull(df_trend["SLOP_x"]) | (pd.isnull(df_trend["SLOP_y"]))]
        #print(df_nan)
        #df_check = pd.merge(df_nan,df_final, how="inner",on=["SYMBOL","PROCDT"])
        #print(df_check[["SYMBOL","PROCDT","SUPPORT_x","SUPPORT_y","RESISTANCE_x","RESISTANCE_y"]])

        df_final.loc[:, "OPEN_S1"] = (df_final["OPEN"] - df_final["SUPPORT1"]) / df_final["SUPPORT1"]
        df_final.loc[:, "CLOSE_S1"] = (df_final["CLOSE"] - df_final["SUPPORT1"]) / df_final["SUPPORT1"]
        df_final.loc[:, "HIGH_S1"] = (df_final["HIGH"] - df_final["SUPPORT1"]) / df_final["SUPPORT1"]
        df_final.loc[:, "LOW_S1"] = (df_final["LOW"] - df_final["SUPPORT1"]) / df_final["SUPPORT1"]
        df_final.loc[:, "OPEN_R1"] = (df_final["OPEN"] - df_final["RESISTANCE1"]) / df_final["RESISTANCE1"]
        df_final.loc[:, "CLOSE_R1"] = (df_final["CLOSE"] - df_final["RESISTANCE1"]) / df_final["RESISTANCE1"]
        df_final.loc[:, "HIGH_R1"] = (df_final["HIGH"] - df_final["RESISTANCE1"]) / df_final["RESISTANCE1"]
        df_final.loc[:, "LOW_R1"] = (df_final["LOW"] - df_final["RESISTANCE1"]) / df_final["RESISTANCE1"]

        df_final.loc[:, "OPEN_S2"] = (df_final["OPEN"] - df_final["SUPPORT2"]) / df_final["SUPPORT2"]
        df_final.loc[:, "CLOSE_S2"] = (df_final["CLOSE"] - df_final["SUPPORT2"]) / df_final["SUPPORT2"]
        df_final.loc[:, "HIGH_S2"] = (df_final["HIGH"] - df_final["SUPPORT2"]) / df_final["SUPPORT2"]
        df_final.loc[:, "LOW_S2"] = (df_final["LOW"] - df_final["SUPPORT2"]) / df_final["SUPPORT2"]
        df_final.loc[:, "OPEN_R2"] = (df_final["OPEN"] - df_final["RESISTANCE2"]) / df_final["RESISTANCE2"]
        df_final.loc[:, "CLOSE_R2"] = (df_final["CLOSE"] - df_final["RESISTANCE2"]) / df_final["RESISTANCE2"]
        df_final.loc[:, "HIGH_R2"] = (df_final["HIGH"] - df_final["RESISTANCE2"]) / df_final["RESISTANCE2"]
        df_final.loc[:, "LOW_R2"] = (df_final["LOW"] - df_final["RESISTANCE2"]) / df_final["RESISTANCE2"]

        df_final = df_final.fillna(0)
        df_final = df_final.replace(0, 0.0001)


        #df_final["SLOP_y"] = abs(df_final["SLOP_y"])
        #df_final=df_bhav
        #print (df_final[["SLOP_x","SLOP_y"]])
        #df_final = df_final[["SYMBOL", "PROCDT", "OPEN", "HIGH", "LOW", "CLOSE", "SLOP_x", "SLOP_y", "TIMESTAMP"]]
        #df_returns = df_final[[key for key in dict(df_final.dtypes) if dict(df_final.dtypes)[key] in ['float64', 'int64']]].pct_change()
        #print(df_returns)


        df_final = df_final [["TIMESTAMP","SYMBOL","CLOSE","MGAP","SSTO_D","SSTO_K","AVGP","HIGH","LOW","OPEN","MINPRICE",
                              "CUMSUM","PUTCALL_RATIO","AB_RATIO_PE","AB_RATIO_CE","AA_RATIO","BB_RATIO",
                              "OPEN_S1","CLOSE_S1","OPEN_R1","CLOSE_R1","OPEN_S2","CLOSE_S2","OPEN_R2","CLOSE_R2","EMA_12","EMA_26",
                              "SLOP_1_UP","SLOP_1_DN","SLOP_2_UP","SLOP_2_DN","SUPPORT1","RESISTANCE1","SUPPORT2","RESISTANCE2"
                              #"EMA20_SLOP1","EMA20_SLOP2","EMA20","EMA50","EMA20_30M","EMA50_30M","EMA20_SLOP1_30M","EMA20_SLOP2_30M"
                              ]]
                              #"EMA20","EMA50","EMA100","EMA20_30M","EMA50_30M","EMA100_30M","EMA20_SLOP1","SLOP2","SLOP1_30M","SLOP2_30M"]]
        df_final= df_final.sort_values(['SYMBOL', 'TIMESTAMP'], ascending=[True, True])
        #print(df_final)
        #df_final.to_csv("c:/tmp/new.txt")

        #df_returns = df_final[[key for key in dict(df_final.dtypes) if dict(df_final.dtypes)[key] in ['float64', 'int64']]].pct_change()


        #print(df_final[(df_final["SYMBOL"] == "NHPC")])
        df_final = df_final[(df_final["SYMBOL"] !="NHPC")]
        #df_final = df_final[(df_final["SYMBOL"] != "BEL")]
        #df_returns["SYMBOL"] = df_final["SYMBOL"]
        #df_returns["TIMESTAMP"] = df_final["TIMESTAMP"]


        symbols=df_final.SYMBOL.unique()
        #df_returns = df_returns[(df_returns["SYMBOL"]=="ITC")]
        #print(df_returns)

        features = []
        output = []
        outputclass=[]
        Data_final= None
        j=0
        for symbol in  symbols:

                Data=df_final[(df_final["SYMBOL"] == symbol)]


                #print(Data)
                Data_ratio = Data[[key for key in dict(Data.dtypes) if dict(Data.dtypes)[key] in ['float64', 'int64']]].pct_change()
                Data_ratio["SYMBOL"]=Data["SYMBOL"]
                Data_ratio["TIMESTAMP"] =Data["TIMESTAMP"]
                Data_ratio["CLV"]  = Data["CLOSE"]


                Data_ratio.loc[Data_ratio["SLOP_1_UP"] < 0, "R1_BROKEN"] = 1
                Data_ratio.loc[Data_ratio["SLOP_2_UP"] < 0, "R2_BROKEN"] = 1
                Data_ratio.loc[Data_ratio["SLOP_1_DN"] > 0, "S1_BROKEN"] = 1
                Data_ratio.loc[Data_ratio["SLOP_2_DN"] > 0, "S2_BROKEN"] = 1

                #print(Data_ratio)
                Data_ratio = Data_ratio.fillna(0)
                Data_ratio = Data_ratio.replace(0, 0.0001)
                #print(Data_ratio)

                # df_returns = df_final[[key for key in dict(df_final.dtypes) if dict(df_final.dtypes)[key] in ['float64', 'int64']]].pct_change()

                F1 = np.array((Data["EMA_12"]))
                #F1 = np.array((Data["EMA20_30M"]))
                #F1 = np.array(Data["SLOP1_30M"])
                #F2 = np.array((Data["EMA50_30M"]))
                F2 = np.array(Data["EMA_26"])
                #F3 = np.array(Data["SSTO_D"])
                F3 = np.array(Data["CLOSE"])

                #F6 = np.array(Data["SLOP_2_UP"])
                #F7= np.array(Data["SLOP_2_DN"])
                F4 = np.array(Data["RESISTANCE1"])
                F5 = np.array(Data["RESISTANCE2"])
                F6 = np.array(Data["SUPPORT1"])
                F7 = np.array(Data["SUPPORT2"])

                #F5 = np.array(Data["SLOP2_30M"])
                #F6 = np.array(Data["SLOP1"])
                #F7 = np.array(Data["SLOP2"])
                #F8 = np.array(Data["EMA20"])
                #F9 = np.array(Data["EMA50"])
                #F10 = np.array(Data["EMA20_30M"])
                #F11 = np.array(Data["EMA50_30M"])
                #F10 = np.array(Data["EMA100"])
                #F9 = np.array(Data["LOW"])
                F12 = np.array(Data["OPEN"])
                #F11 = np.array(Data["CLOSE"])
                #F12 = np.array(Data["AVGP"])
                #F13 = np.array(Data["SLOP1"])
                #F14 = np.array(Data["SLOP2"])
                F13 = np.array(Data["SLOP_1_UP"])
                F14 = np.array(Data["SLOP_1_DN"])
                F15 = np.array(Data["SLOP_2_UP"])
                F16 = np.array(Data["SLOP_2_DN"])
                #F17 = np.array(Data["EMA20_SLOP1_30M"])
                #F18 = np.array(Data["EMA20_SLOP2_30M"])
                #F19 = np.array(Data["EMA20_SLOP1"])
                #F20 = np.array(Data["EMA20_SLOP2"])
                F21 = np.array((Data["PUTCALL_RATIO"]))
                F13 = np.array(Data["AB_RATIO_PE"])
                F14 = np.array(Data["AB_RATIO_CE"])
                F15 = np.array(Data["AA_RATIO"])
                F16 = np.array(Data["BB_RATIO"])
                #F11 = np.array(Data["SSTO_D"])

                #F8 = np.array(Data["CLOSE_R2"])
                #F9 = np.array(Data["OPEN_R2"])
                #F10 = np.array(Data["CLOSE_S2"])
                #F11 = np.array(Data["OPEN_S2"])

                """
                F1=np.array((Data_ratio["CUMSUM"] ))
                #F1=np.array(Data_ratio["CLOSE"])
                #F2 = np.array(Data["PUTCALL_RATIO"])
                F2 = np.array(Data["AB_RATIO_PE"])
                F3 = np.array(Data["AB_RATIO_CE"])
                F4 = np.array(Data["AA_RATIO"])
                F5=  np.array(Data["BB_RATIO"])
                F6 = np.array(Data["MGAP"])
                F7 = np.array(Data["SSTO_D"])
                #F8 = np.array(Data_ratio["SLOP_1_UP"])
                #F9 = np.array(Data_ratio["SLOP_1_DN"])
                F8 = np.array(Data_ratio["SLOP_1_UP"])
                F9 = np.array(Data_ratio["SLOP_1_DN"])
                F10 = np.array(Data_ratio["SLOP_2_UP"])
                F11 = np.array(Data_ratio["SLOP_2_DN"])
                F12 = np.array(Data_ratio["R1_BROKEN"])
                F13 = np.array(Data_ratio["S1_BROKEN"])
                F14 = np.array(Data_ratio["R2_BROKEN"])
                F15 = np.array(Data_ratio["S2_BROKEN"])
                #F12 = np.array(Data["CLOSE_R1"])
                #F13 = np.array(Data["OPEN_R1"])
                #F14 = np.array(Data["CLOSE_S1"])
                #F15 = np.array(Data["OPEN_S1"])
                F20 = np.array(Data_ratio["CLOSE"])
                """

                #print(xData)
                yData=np.array(Data_ratio["CLOSE"])
                #xData = np.array((Data["Macd"]))
                if Train:
                        featuresize = len(Data["CLOSE"])-10
                       # Data_final =  pd.concat([Data_final, Data_ratio.iloc[5:,:]])
                        Data_final = pd.concat([Data_final, Data_ratio.iloc[11:, :]])
                else:
                        featuresize =len(Data["CLOSE"])-9
                        Data_final = pd.concat([Data_final, Data_ratio.iloc[10:, :]])

                for i in range(2, featuresize):
                        tmpfeatures = []
                        tmpoutput = []
                        tmpoutputclass = []

                        """tup = [ F1[i], F1[i+1], F1[i+2], F1[i+3], F1[i+4], F1[i+5], F1[i+6], F1[i+7], F1[i+8],F1[i+9],F1[i+10],
                                F2[i], F2[i + 1], F2[i + 2], F2[i + 3], F2[i + 4], F2[i + 5], F2[i + 6], F2[i + 7],F2[i + 8],F2[i+9],F2[i+10],
                                F3[i], F3[i + 1], F3[i + 2], F3[i + 3], F3[i + 4], F3[i + 5], F3[i + 6], F3[i + 7],F3[i + 8],F3[i+9],F3[i+10],
                                F4[i], F4[i + 1], F4[i + 2], F4[i + 3], F4[i + 4], F4[i + 5], F4[i + 6], F4[i + 7],F4[i + 8],F4[i+9],F4[i+10],
                                F5[i], F5[i + 1], F5[i + 2], F5[i + 3], F5[i + 4], F5[i + 5], F5[i + 6], F5[i + 7],F5[i + 8],F5[i+9],F5[i+10],
                                F6[i], F6[i + 1], F6[i + 2], F6[i + 3], F6[i + 4], F6[i + 5], F6[i + 6], F6[i + 7],F6[i + 8],F6[i+9],F6[i+10],
                                F7[i], F7[i + 1], F7[i + 2], F7[i + 3], F7[i + 4], F7[i + 5], F7[i + 6], F7[i + 7],F7[i + 8],F7[i+9],F7[i+10],
                                F8[i], F8[i + 1], F8[i + 2], F8[i + 3], F8[i + 4], F8[i + 5], F8[i + 6], F8[i + 7],F8[i + 8],F8[i+9],F8[i+10],
                                F9[i], F9[i + 1], F9[i + 2], F9[i + 3], F9[i + 4], F9[i + 5], F9[i + 6], F9[i + 7],F9[i + 8],F9[i+9],F9[i+10],
                                F10[i], F10[i + 1], F10[i + 2], F10[i + 3], F10[i + 4], F10[i + 5], F10[i + 6], F10[i + 7],F10[i + 8],F10[i+9],F10[i+10]]
                                #F11[i], F11[i + 1], F11[i + 2], F11[i + 3], F11[i + 4], F11[i + 5], F11[i + 6], F11[i + 7],F11[i + 8],F11[i+9],F11[i+10]]
                                """
                        """
                        if F10[i] >= F11[i]:
                                dir = 1
                        else:
                                dir = 0
                                """

                        tup = [
                                 F13[i + 3], F14[i + 3], F15[i + 3], F16[i + 3],F21[i+3]  # F10[i+8],
                                ]

                        tmpfeatures.append(tup)
                        if Train:
                                tmpoutput.append([yData[i + 4]])
                                OneHot = GetOneHot(yData[i + 4] * 100, LableDict, ZeroVec)
                                tmpoutputclass.append(OneHot)
                                """
                                if yData[i+9] >= yData[i+8] :
                                        outputclass.append([True])
                                else:
                                        outputclass.append([False])"""


                        features.append(tmpfeatures[(len(tmpfeatures)-len(tmpfeatures)%20)])
                        output.append(tmpoutput[(len(tmpoutput) - len(tmpoutput) % 20)])
                        outputclass.append(tmpoutputclass[(len(tmpoutputclass) - len(tmpoutputclass) % 20)])
        # returns["Nifty"] = df_nify["Nifty"]
        #print(features)
        #np.savetxt("c:/tmp/val.txt",features)
        #np.savetxt("c:/tmp/out.txt",outputclass)
                Data_final=Data_final[["TIMESTAMP","SYMBOL","CLV","CLOSE"]]
        #print(output)
        #print(Data_final)
        return (   np.array(features), np.array(output), np.array(outputclass),Data_final.reset_index())





def  Parse_data_asarray():
        data="C:/tmp/transactions.csv.txt"
        df_nify=pd.read_csv(data,sep=",",usecols=[0,3],names=['Date','Nifty'],header=-1)
        df_nf = pd.read_csv(data, sep=",", usecols=[0, 3], names=['Date', 'Ncol'], header=-1)
        df_nify['Ncol']=df_nf['Ncol']
     #   print(df_nify)
     #   df_nify['Date']=pd.to_datetime(df_nify['Date'],format='%Y-%m-%d')
        df_nify['Date'] = pd.to_numeric(df_nify['Date'])
        df_nify = df_nify.sort_values(['Date'], ascending=[True])
        returns = df_nify[[key for key in dict(df_nify.dtypes) if dict(df_nify.dtypes)[key] in ['float64', 'int64']]]\
                     .pct_change()

        xData = np.array(returns['Date'])[1:]
        yData = np.array(returns['Date'])[1:]
    #    print(xData)
     #   ndata = xData.reshape(1, -1)
      #  print(ndata)
        return (xData, yData)




pd.set_option('display.height',500)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',200)
#pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_columns',50)
"""a,z = SetLabels()
for i in range(8):
        print(i)
        oneh= GetOneHot((i)*-1.0,a,z)
        idx=oneh.index(1)
        print(idx)
        print(a)
"""

#Parse_data(True)
#Parse_intraday()
#Parse_mum(True)
#df=Parse_intraday()
#print(df)
#(x,y,z,df)=Parse_mum(True)
#(x,y, z)=Parse_mum()
#Parse_data(True)
#print(x.shape)
#print(x)
#print(y.shape)
#print(y)
#print(z)
#print(z.size)
#print(z.shape)


#print(np.array(features))
#print(np.array(output).reshape(-1,1))

#(x, y)= Parse_data()
#combined = np.vstack((x, y)).T
#print(combined)
#print(x)
#print(y)
