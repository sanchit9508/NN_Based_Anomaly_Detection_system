import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import random

def synthesis_data(df,cloud):
    month_range = np.arange(1,13,1)
    date = datetime.now()
    year_limit = date.year
    month_limit = date.month
    synthetic_data_final = pd.DataFrame()
    for i in [2024,2025]:
        for j in month_range:
            if i<=year_limit and j<=month_limit:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df)
                epochs_list=[50,100,200,400]
                batch_size_list = [500,750,1000]
                epoch = random.choice(epochs_list)
                batch_size = random.choice(batch_size_list)
                print(epoch,batch_size)
                synthesizer = CTGANSynthesizer(metadata,epochs=epoch,batch_size=batch_size,cuda=True)
                synthesizer.fit(df)
                if cloud=='AWS':
                    synthetic_data = synthesizer.sample(num_rows=int(len(df)*2))
                elif cloud=='AZURE':
                    synthetic_data = synthesizer.sample(num_rows=len(df)*6)
                else:
                    synthetic_data = synthesizer.sample(num_rows=len(df))
                synthetic_data[['chargeperiodstart_year','chargeperiodend_year', 
                                'billingperiodstart_year','billingperiodend_year']]=i
                synthetic_data[['billingperiodend_month','billingperiodstart_month', 
                                'chargeperiodstart_month','chargeperiodend_month']]=j
                synthetic_data_final = pd.concat([synthetic_data_final,synthetic_data])
                print("Size_batch:",synthetic_data_final.shape)
            else:
                break
    print("Data Shape",synthetic_data_final.shape)
    return synthetic_data_final