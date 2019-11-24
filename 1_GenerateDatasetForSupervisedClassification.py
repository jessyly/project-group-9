"""
 Task: 
     Predict future likely medical condition from the medical history of a patient
     
 Approach:
     Supervised classification using Decision Trees, RandomForests, and XGBoost
     Target Label: Last occurrence of disease
     Features: One-hot encoded features of first occurrence of each of distinct 
               disease and other features as specified (could be medicines, 
               atclevel3code, atclevel4code, atclevel5code) from patient history 
               merged into a single data frame with one record per patient  
     Validation: 5-fold cross-validation on 80% training data 
     Final test score: Using held-out 20% test data
"""

import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import snowflake.connector as sf

os.chdir(r'C:\\FINAL')

"""
 Specify what type of one-hot features to create.
 Possible values are: 'disease', 'medicine', 'atclevel3code', 'atclevel4code', 
                      'atclevel5code', 'days_lag_disease', 'days_lag_medicine',
                      'days_lag_atclevel3code', 'days_lag_atclevel4code', 
                      'days_lag_atclevel5code'
"""
feature_type = ['disease', 'medicine', 'atclevel3code', 'atclevel4code',
                'atclevel5code', 'days_lag_disease', 'days_lag_medicine',
                'days_lag_atclevel3code', 'days_lag_atclevel4code', 
                'days_lag_atclevel5code'] 

"""
 Construct 'train', keeping only the 'MasterPatientID' and the record of first 
 occurrence of the last disease for each patient (which will be the target label)
 Also, duplicate MasterPatientID rows having multiple target labels (i.e. multiple 
 diseases contracted on the same last date of prescription history), one for each target, 
 to cater for multilabel classification.
 This 'train' will then be merged with other features as they are built
"""
try: 
    # Open connection with Snowflake database
    sf_conn = sf.connect(user='******', password='******',
                         account='******')

    #sf_conn = engine.connect()
    
    # Read patient disease history from MySQL database       
    nd_patient_history_disease = pd.read_sql("select \
                           MASTERPATIENTID \
                    	 , PBSDISEASEGROUP \
                         , 1 DISEASE_PRESENCE \
                         , DAYSLAGDISNO \
                         , DAYSLAG_DIS_PREVDIS \
                         , DISPENSECALENDARDATE \
                         from \
                         ( \
                        	 select distinct '_DUMMY' MASTERPATIENTID \
                        		 , '1900-01-02' DISPENSECALENDARDATE \
                        		 ,  concat('DIS_', DISEASE_GROUP) PBSDISEASEGROUP \
     							 ,  '1' DAYSLAGDISNO \
                            	 ,  0 DAYSLAG_DIS_PREVDIS \
                        	 from ppo.pbscodetodiseaseconditionmapping \
                        	 UNION \
                        	 select \
                        	      MASTERPATIENTID \
                        	  	 , DISPENSECALENDARDATE \
                        	     , concat('DIS_', PBSDISEASEGROUP) PBSDISEASEGROUP \
                                 , concat('DISLAG_', row_number() over (partition by MASTERPATIENTID order by DISPENSECALENDARDATE asc)) DAYSLAGDISNO \
                                 , DAYSLAG_DIS_PREVDIS \
                        	     from ppo.nd_patient_history_distinct_diseases_final_20191020 \
                         ) a \
                         order by MASTERPATIENTID, DISPENSECALENDARDATE, \
                                  PBSDISEASEGROUP;" 
                        , con=sf_conn)   

except Exception as e:
    print(e)
    
tmp = nd_patient_history_disease.groupby(['MASTERPATIENTID']). \
                            agg({'DISPENSECALENDARDATE':np.max}).reset_index()
train = pd.merge(nd_patient_history_disease, tmp, how='inner', 
                 on=['MASTERPATIENTID', 'DISPENSECALENDARDATE'])

# Drop '_DUMMY' patient
train = train[train.MASTERPATIENTID!='_DUMMY']

# Drop unnecessary columns
train = train.drop(['DISEASE_PRESENCE', 'DISPENSECALENDARDATE', 'DAYSLAGDISNO',
                    'DAYSLAG_DIS_PREVDIS'], axis=1)

# Rename target label column
train = train.rename(columns={'PBSDISEASEGROUP': 'TARGET_LABEL'})

# ------------------------------------------------------------------------
#							CREATE FEATURES
# ------------------------------------------------------------------------
# Remove from features the row corresponding to target label (last occurrence of disease)
nd_patient_history_disease = pd.merge(nd_patient_history_disease, tmp, 
                                          how='inner', on=['MASTERPATIENTID'])
nd_patient_history_disease = pd.concat([nd_patient_history_disease
                    [nd_patient_history_disease.MASTERPATIENTID=='_DUMMY'], 
                    nd_patient_history_disease[nd_patient_history_disease
                    ['DISPENSECALENDARDATE_x'] < nd_patient_history_disease
                    ['DISPENSECALENDARDATE_y']]])
    
# Create one-hot features based on disease from prescription history
if 'disease' in feature_type:   
    # Drop unnecessary columns
    x_nd_patient_history_disease = nd_patient_history_disease.drop(
                    ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y', 
                     'DAYSLAGDISNO', 'DAYSLAG_DIS_PREVDIS'], axis=1)
       
    # Pivot rows into columns to have one record per patient
    x_nd_patient_history_disease = x_nd_patient_history_disease.pivot(
                                                     index='MASTERPATIENTID' 
                                                   , columns='PBSDISEASEGROUP') 
    x_nd_patient_history_disease.reset_index(level=0, inplace=True)
    
    # Convert multiindex column names to single index column names
    x_nd_patient_history_disease.columns = [b if b else a for a, b in 
                                            x_nd_patient_history_disease.columns]
    
    # Drop '_DUMMY' patient added for creating one-hot columns for all diseases
    x_nd_patient_history_disease = x_nd_patient_history_disease[
                        x_nd_patient_history_disease.MASTERPATIENTID!='_DUMMY']

    # Replace null values with 0    
    x_nd_patient_history_disease.fillna(0, inplace=True)
    
    train = pd.merge(train, x_nd_patient_history_disease, how='inner',
                     on='MASTERPATIENTID')
    
    del x_nd_patient_history_disease

if 'days_lag_disease' in feature_type:   
    # Drop unnecessary columns
    x_nd_patient_history_disease = nd_patient_history_disease.drop(
                    ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                     'PBSDISEASEGROUP', 'DISEASE_PRESENCE'], axis=1)
        
    # Drop '_DUMMY' patient added for creating one-hot columns for all diseases
    x_nd_patient_history_disease = x_nd_patient_history_disease[
                        x_nd_patient_history_disease.MASTERPATIENTID!='_DUMMY']
       
    # Pivot rows into columns to have one record per patient
    x_nd_patient_history_disease = x_nd_patient_history_disease.pivot(
                                                     index='MASTERPATIENTID' 
                                                   , columns='DAYSLAGDISNO') 
    x_nd_patient_history_disease.reset_index(level=0, inplace=True)
    
    # Convert multiindex column names to single index column names
    x_nd_patient_history_disease.columns = [b if b else a for a, b in 
                                            x_nd_patient_history_disease.columns]
    
    # Replace null values with -1    
    x_nd_patient_history_disease.fillna(-1, inplace=True)
    
    train = pd.merge(train, x_nd_patient_history_disease, how='inner', 
                     on='MASTERPATIENTID')
    
    del x_nd_patient_history_disease
      
del nd_patient_history_disease

# Create one-hot features based on disease from prescription history
if ('medicine' in feature_type) or ('days_lag_medicine' in feature_type):
    try: 
        # Read patient medicine history from MySQL database       
        nd_patient_history_medicine = pd.read_sql("select \
                               MASTERPATIENTID \
                        	 , GENERICINGREDIENTNAME \
                             , 1 MEDICINE_PRESENCE \
                             , DAYSLAGMEDNO \
                             , DAYSLAG_MED_PREVMED \
                             , DISPENSECALENDARDATE \
                             from \
                             ( \
                            	 select distinct '_DUMMY' MASTERPATIENTID \
                            		 , '1900-01-01' DISPENSECALENDARDATE \
                            		 ,  concat('MED_', GENERICINGREDIENTNAME) GENERICINGREDIENTNAME \
                                     ,  '1' DAYSLAGMEDNO \
                                     , 0 DAYSLAG_MED_PREVMED \
                            	 from ppo.dimproductmaster \
                            	 UNION \
                            	 select \
                            	      MASTERPATIENTID \
                            	  	 , DISPENSECALENDARDATE \
                            	     , concat('MED_', GENERICINGREDIENTNAME) GENERICINGREDIENTNAME \
                                     , concat('MEDLAG_', row_number() over \
                                         (partition by MASTERPATIENTID \
                                         order by DISPENSECALENDARDATE asc)) DAYSLAGMEDNO \
                                     , DAYSLAG_MED_PREVMED \
                            	     from ppo.nd_patient_history_distinct_medicines_final_20191020 \
                             ) a \
                             order by MASTERPATIENTID, DISPENSECALENDARDATE, \
                                      GENERICINGREDIENTNAME;"
                             , con=sf_conn)   

    except Exception as e:
        print(e)
        
    """
     Remove from features all medicine history which is on or after the first 
     occurrence of last disease (target label) i.e. we cannot use as features 
     any subsequent medicine which the patient took while or after being affected  
     with the target disease that we are trying to predict 
    """
    nd_patient_history_medicine = pd.merge(nd_patient_history_medicine, tmp,
                                              how='inner', on=['MASTERPATIENTID'])
    nd_patient_history_medicine = nd_patient_history_medicine[
                    nd_patient_history_medicine['DISPENSECALENDARDATE_x'] <
                    nd_patient_history_medicine['DISPENSECALENDARDATE_y']]
    
    if 'medicine' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_medicine = nd_patient_history_medicine.drop(
                       ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                        'DAYSLAGMEDNO', 'DAYSLAG_MED_PREVMED'], axis=1)
           
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_medicine = x_nd_patient_history_medicine.pivot(
                                                        index='MASTERPATIENTID'
                                                      , columns='GENERICINGREDIENTNAME') 
        x_nd_patient_history_medicine.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_medicine.columns = [b if b else a for a, b in
                                                 x_nd_patient_history_medicine.columns]
        
        # Drop '_DUMMY' patient added for creating one-hot columns for all medicines
        x_nd_patient_history_medicine = x_nd_patient_history_medicine[
                          x_nd_patient_history_medicine.MASTERPATIENTID!='_DUMMY']

        # Replace null values with 0    
        x_nd_patient_history_medicine.fillna(0, inplace=True)
        
        train = pd.merge(train, x_nd_patient_history_medicine, how='inner',
                         on='MASTERPATIENTID')
               
        del x_nd_patient_history_medicine

    if 'days_lag_medicine' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_medicine = nd_patient_history_medicine.drop(
                       ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                        'GENERICINGREDIENTNAME', 'MEDICINE_PRESENCE'], axis=1)

        # Drop '_DUMMY' patient added for creating one-hot columns for all medicines
        x_nd_patient_history_medicine = x_nd_patient_history_medicine[
                          x_nd_patient_history_medicine.MASTERPATIENTID!='_DUMMY']
           
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_medicine = x_nd_patient_history_medicine.pivot(
                                                        index='MASTERPATIENTID'
                                                      , columns='DAYSLAGMEDNO') 
        x_nd_patient_history_medicine.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_medicine.columns = [b if b else a for a, b in
                                                 x_nd_patient_history_medicine.columns]

        # Replace null values with -1    
        x_nd_patient_history_medicine.fillna(-1, inplace=True)
            
        train = pd.merge(train, x_nd_patient_history_medicine, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_medicine   
        
    del nd_patient_history_medicine 

if ('atclevel3code' in feature_type) or ('days_lag_atclevel3code' in feature_type):    
    try:
        # Read patient medicine classification (ATCLevel3Code) history from MySQL database       
        nd_patient_history_atclevel3code = pd.read_sql("select \
                               MASTERPATIENTID \
                        	 , ATCLEVEL3CODE \
                             , 1 ATC3_PRESENCE \
                             , DAYSLAGATC3NO \
                             , DAYSLAG_ATC_PREVATC \
                             , DISPENSECALENDARDATE \
                             from \
                             ( \
                            	 select distinct '_DUMMY' MASTERPATIENTID \
                            		 , '1900-01-01' DISPENSECALENDARDATE \
                            		 ,  concat('ATC3_', ATCLEVEL3CODE) ATCLEVEL3CODE \
                                     ,  '1' DAYSLAGATC3NO \
                                     ,  0 DAYSLAG_ATC_PREVATC \
                            	 from ppo.dimproductmaster \
                            	 UNION \
                            	 select \
                            	      MASTERPATIENTID \
                            	  	 , DISPENSECALENDARDATE \
                            	     , concat('ATC3_', ATCLEVEL3CODE) ATCLEVEL3CODE \
                                     , concat('ATC3LAG_', row_number() over ( \
                                       partition by MASTERPATIENTID \
                                       order by DISPENSECALENDARDATE asc)) DAYSLAGATC3NO \
                                     , DAYSLAG_ATC_PREVATC \
                        	     from ppo.nd_patient_history_distinct_atclevel3code_final_20191020 \
                             ) a \
                             order by MASTERPATIENTID, DISPENSECALENDARDATE, \
                                      ATCLEVEL3CODE;"
                             , con=sf_conn)   

    except Exception as e:
        print(e)

    """
     Remove from features all medicine classification (ATCLevel3Code) history 
     which is on or after the first occurrence of last disease (target label) 
     i.e. we cannot use as features any subsequent medicine classification 
     (ATCLevel3Code) which the patient took while or after being affected with 
     the target disease that we are trying to predict 
    """
    nd_patient_history_atclevel3code = pd.merge(nd_patient_history_atclevel3code, tmp,
                                              how='inner', on=['MASTERPATIENTID'])
    nd_patient_history_atclevel3code = nd_patient_history_atclevel3code[
                    nd_patient_history_atclevel3code['DISPENSECALENDARDATE_x'] <
                    nd_patient_history_atclevel3code['DISPENSECALENDARDATE_y']]

    if 'atclevel3code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel3code = nd_patient_history_atclevel3code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                      'DAYSLAGATC3NO', 'DAYSLAG_ATC_PREVATC'], axis=1)
           
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel3code = x_nd_patient_history_atclevel3code.pivot(
                                                          index='MASTERPATIENTID'
                                                        , columns='ATCLEVEL3CODE') 
        x_nd_patient_history_atclevel3code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel3code.columns = [b if b else a for a, b in
                                          x_nd_patient_history_atclevel3code.columns]
        
        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel3Codes
        x_nd_patient_history_atclevel3code = x_nd_patient_history_atclevel3code[
                    x_nd_patient_history_atclevel3code.MASTERPATIENTID!='_DUMMY']
    
        # Replace null values with 0    
        x_nd_patient_history_atclevel3code.fillna(0, inplace=True)
            
        train = pd.merge(train, x_nd_patient_history_atclevel3code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel3code

    if 'days_lag_atclevel3code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel3code = nd_patient_history_atclevel3code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                      'ATCLEVEL3CODE', 'ATC3_PRESENCE'], axis=1)

        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel3Codes
        x_nd_patient_history_atclevel3code = x_nd_patient_history_atclevel3code[
                    x_nd_patient_history_atclevel3code.MASTERPATIENTID!='_DUMMY']
                    
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel3code = x_nd_patient_history_atclevel3code.pivot(
                                                          index='MASTERPATIENTID'
                                                        , columns='DAYSLAGATC3NO') 
        x_nd_patient_history_atclevel3code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel3code.columns = [b if b else a for a, b in
                                          x_nd_patient_history_atclevel3code.columns]
            
        # Replace null values with -1    
        x_nd_patient_history_atclevel3code.fillna(-1, inplace=True)
            
        train = pd.merge(train, x_nd_patient_history_atclevel3code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel3code
        
    del nd_patient_history_atclevel3code
        
if ('atclevel4code' in feature_type) or ('days_lag_atclevel4code' in feature_type):    
    try:
        # Read patient medicine classification (ATCLevel3Code) history from MySQL database       
        nd_patient_history_atclevel4code = pd.read_sql("select \
                               MASTERPATIENTID \
                        	 , ATCLEVEL4CODE \
                             , 1 ATC4_PRESENCE \
                             , DAYSLAGATC4NO \
                             , DAYSLAG_ATC_PREVATC  \
                             , DISPENSECALENDARDATE \
                             from \
                             ( \
                            	 select distinct '_DUMMY' MASTERPATIENTID \
                            		 , '1900-01-01' DISPENSECALENDARDATE \
                            		 ,  concat('ATC4_', ATCLEVEL4CODE) ATCLEVEL4CODE \
                                     ,  '1' DAYSLAGATC4NO \
                                     ,  0 DAYSLAG_ATC_PREVATC  \
                            	 from ppo.dimproductmaster \
                            	 UNION \
                            	 select \
                            	      MASTERPATIENTID \
                            	  	 , DISPENSECALENDARDATE \
                            	     , concat('ATC4_', ATCLEVEL4CODE) ATCLEVEL4CODE \
                                     , concat('ATC4LAG_', row_number() over ( \
                                       partition by MASTERPATIENTID \
                                       order by DISPENSECALENDARDATE asc)) DAYSLAGATC4NO \
                                     , DAYSLAG_ATC_PREVATC \
                        	     from ppo.nd_patient_history_distinct_atclevel4code_final_20191020 \
                             ) a \
                             order by MASTERPATIENTID, DISPENSECALENDARDATE, \
                                      ATCLEVEL4CODE;"
                             , con=sf_conn)   

    except Exception as e:
        print(e)

    """
     Remove from features all medicine classification (ATCLevel4Code) history 
     which is on or after the first occurrence of last disease (target label) 
     i.e. we cannot use as features any subsequent medicine classification 
     (ATCLevel4Code) which the patient took while or after being affected with 
     the target disease that we are trying to predict 
    """
    nd_patient_history_atclevel4code = pd.merge(nd_patient_history_atclevel4code, tmp,
                                              how='inner', on=['MASTERPATIENTID'])
    nd_patient_history_atclevel4code = nd_patient_history_atclevel4code[
                    nd_patient_history_atclevel4code['DISPENSECALENDARDATE_x'] <
                    nd_patient_history_atclevel4code['DISPENSECALENDARDATE_y']]

    if 'atclevel4code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel4code = nd_patient_history_atclevel4code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y', 
                      'DAYSLAGATC4NO', 'DAYSLAG_ATC_PREVATC'], axis=1)
           
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel4code = x_nd_patient_history_atclevel4code.pivot(
                                                          index='MASTERPATIENTID'
                                                        , columns='ATCLEVEL4CODE') 
        x_nd_patient_history_atclevel4code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel4code.columns = [b if b else a for a, b in
                                      x_nd_patient_history_atclevel4code.columns]
        
        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel4Codes
        x_nd_patient_history_atclevel4code = x_nd_patient_history_atclevel4code[
                    x_nd_patient_history_atclevel4code.MASTERPATIENTID!='_DUMMY']
    
        # Replace null values with 0    
        x_nd_patient_history_atclevel4code.fillna(0, inplace=True)
        
        train = pd.merge(train, x_nd_patient_history_atclevel4code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel4code

    if 'days_lag_atclevel4code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel4code = nd_patient_history_atclevel4code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y', 
                      'ATCLEVEL4CODE', 'ATC4_PRESENCE'], axis=1)

        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel4Codes
        x_nd_patient_history_atclevel4code = x_nd_patient_history_atclevel4code[
                    x_nd_patient_history_atclevel4code.MASTERPATIENTID!='_DUMMY']
               
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel4code = x_nd_patient_history_atclevel4code.pivot(
                                                          index='MASTERPATIENTID' 
                                                        , columns='DAYSLAGATC4NO') 
        x_nd_patient_history_atclevel4code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel4code.columns = [b if b else a for a, b in
                                      x_nd_patient_history_atclevel4code.columns]
        
        # Replace null values with -1    
        x_nd_patient_history_atclevel4code.fillna(-1, inplace=True)
        
        train = pd.merge(train, x_nd_patient_history_atclevel4code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel4code
        
    del nd_patient_history_atclevel4code
        
if ('atclevel5code' in feature_type) or ('days_lag_atclevel5code' in feature_type):    
    try:
        # Read patient medicine classification (ATCLevel3Code) history from MySQL database       
        nd_patient_history_atclevel5code = pd.read_sql("select \
                               MASTERPATIENTID \
                        	 , ATCLEVEL5CODE \
                             , 1 ATC5_PRESENCE \
                             , DAYSLAGATC5NO  \
                             , DAYSLAG_ATC_PREVATC \
                             , DISPENSECALENDARDATE \
                             from \
                             ( \
                            	 select distinct '_DUMMY' MASTERPATIENTID \
                            		 , '1900-01-01' DISPENSECALENDARDATE \
                            		 ,  concat('ATC5_', ATCLEVEL5CODE) ATCLEVEL5CODE \
                                     ,  '1' DAYSLAGATC5NO \
                                     ,  0 DAYSLAG_ATC_PREVATC \
                            	 from ppo.dimproductmaster \
                            	 UNION \
                            	 select \
                            	      MASTERPATIENTID \
                            	  	 , DISPENSECALENDARDATE \
                            	     , concat('ATC5_', ATCLEVEL5CODE) ATCLEVEL5CODE \
                                     , concat('ATC5LAG_', row_number() over ( \
                                       partition by MASTERPATIENTID \
                                       order by DISPENSECALENDARDATE asc)) DAYSLAGATC5NO \
                                     , DAYSLAG_ATC_PREVATC \
                        	     from ppo.nd_patient_history_distinct_atclevel5code_final_20191020 \
                             ) a \
                             order by MASTERPATIENTID, DISPENSECALENDARDATE, \
                                      ATCLEVEL5CODE;"
                             , con=sf_conn)   

    except Exception as e:
        print(e)

    """
     Remove from features all medicine classification (ATCLevel5Code) history 
     which is on or after the first occurrence of last disease (target label) 
     i.e. we cannot use as features any subsequent medicine classification 
     (ATCLevel5Code) which the patient took while or after being affected with 
     the target disease that we are trying to predict 
    """
    nd_patient_history_atclevel5code = pd.merge(nd_patient_history_atclevel5code, tmp,
                                              how='inner', on=['MASTERPATIENTID'])
    nd_patient_history_atclevel5code = nd_patient_history_atclevel5code[
                    nd_patient_history_atclevel5code['DISPENSECALENDARDATE_x'] <
                    nd_patient_history_atclevel5code['DISPENSECALENDARDATE_y']]

    if 'atclevel5code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel5code = nd_patient_history_atclevel5code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                      'DAYSLAGATC5NO', 'DAYSLAG_ATC_PREVATC'], axis=1)
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel5code = x_nd_patient_history_atclevel5code.pivot(
                                                          index='MASTERPATIENTID'
                                                        , columns='ATCLEVEL5CODE') 
        x_nd_patient_history_atclevel5code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel5code.columns = [b if b else a for a, b in
                                      x_nd_patient_history_atclevel5code.columns]
        
        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel5Codes
        x_nd_patient_history_atclevel5code = x_nd_patient_history_atclevel5code[
                      x_nd_patient_history_atclevel5code.MASTERPATIENTID!='_DUMMY']
    
        # Replace null values with 0    
        x_nd_patient_history_atclevel5code.fillna(0, inplace=True)
    
        train = pd.merge(train, x_nd_patient_history_atclevel5code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel5code

    if 'days_lag_atclevel5code' in feature_type:
        # Drop unnecessary columns
        x_nd_patient_history_atclevel5code = nd_patient_history_atclevel5code.drop(
                     ['DISPENSECALENDARDATE_x', 'DISPENSECALENDARDATE_y',
                      'ATCLEVEL5CODE', 'ATC5_PRESENCE'], axis=1)

        # Drop '_DUMMY' patient added for creating one-hot columns for all ATCLevel5Codes
        x_nd_patient_history_atclevel5code = x_nd_patient_history_atclevel5code[
                      x_nd_patient_history_atclevel5code.MASTERPATIENTID!='_DUMMY']   
         
        # Pivot rows into columns to have one record per patient
        x_nd_patient_history_atclevel5code = x_nd_patient_history_atclevel5code.pivot(
                                                          index='MASTERPATIENTID'
                                                        , columns='DAYSLAGATC5NO') 
        x_nd_patient_history_atclevel5code.reset_index(level=0, inplace=True)
        
        # Convert multiindex column names to single index column names
        x_nd_patient_history_atclevel5code.columns = [b if b else a for a, b in
                                      x_nd_patient_history_atclevel5code.columns]
            
        # Replace null values with -1    
        x_nd_patient_history_atclevel5code.fillna(-1, inplace=True)
    
        train = pd.merge(train, x_nd_patient_history_atclevel5code, how='inner',
                         on='MASTERPATIENTID')
        
        del x_nd_patient_history_atclevel5code

    del nd_patient_history_atclevel5code    

sf_conn.close()    

del tmp, feature_type

train.to_csv('full_train_prel.csv', index=False)

# Convert all the target disease labels having master patient records less than 100 to a common label 'DIS_1000'
train_new = train[['TARGET_LABEL', 'MASTERPATIENTID']]
target_label_count = train_new.groupby('TARGET_LABEL').agg(['count'])
target_label_count.columns = [a + '_count' if a else b for a, b in 
                              target_label_count.columns]   # Convert multiindex column names to single index column names
target_label_count.reset_index(level=0, inplace=True)
convert_labels = target_label_count[target_label_count.MASTERPATIENTID_count<100]['TARGET_LABEL']
convert_labels = pd.DataFrame(convert_labels)
convert_labels.to_csv('convert_labels.csv', index=False)    
convert_labels = convert_labels['TARGET_LABEL'].values.tolist()
train.loc[train['TARGET_LABEL'].isin(convert_labels), 'TARGET_LABEL'] = 'DIS_1000'

del train_new, convert_labels, target_label_count

# Get target label counts
target_label_count = train[['TARGET_LABEL', 'MASTERPATIENTID']].groupby('TARGET_LABEL').agg(['count'])
target_label_count.columns = [a + '_count' if a else b for a, b in 
                              target_label_count.columns]           # Convert multiindex column names to single index column names

# Get Top 51 target labels
top51 = target_label_count.sort_values('MASTERPATIENTID_count', ascending = False).head(51)

# Save Top51 target labels and target label counts to csv
top51.reset_index(level=0, inplace=True)
top51.to_csv('top51_target_label_count.csv', index=False)
target_label_count.reset_index(level=0, inplace=True)
target_label_count.to_csv('target_label_count.csv', index=False)

# Only keep patients with target disease in top 51 target labels
train = train[train['TARGET_LABEL'].isin(top51['TARGET_LABEL'])]

# Encode labels with values between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
le.fit(train['TARGET_LABEL'])
train['TARGET_LABEL_ENCODING'] = le.transform(train['TARGET_LABEL']) 
targel_Label_mapping = train[['TARGET_LABEL', 'TARGET_LABEL_ENCODING']].drop_duplicates()
targel_Label_mapping.to_csv('targel_label_mapping.csv', index=False)

del targel_Label_mapping, top51, target_label_count

train['TARGET_LABEL'] = train['TARGET_LABEL_ENCODING']
train = train.drop('TARGET_LABEL_ENCODING', axis=1)

train.to_csv('full_train.csv', index=False)
