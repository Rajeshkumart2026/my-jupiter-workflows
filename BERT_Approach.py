import pandas as pd
import numpy as np
from bertopic import BERTopic
import warnings
warnings.filterwarnings('ignore')

path = r""C:\Users\rajeshkumar.t\Desktop\PL_Outputs.xlsx""
df = pd.read_excel(path)

df['Clean_Text'] = df['Description'].fillna('').str.lower().str.replace('undefined - undefined - ', '', regex=True)
df['CallTypeDetails'] = df['CallTypeDetails'].fillna('').str.lower()

conditions = [
    df['Clean_Text'].str.contains(r'out of (spd|spp)|spd breach|sla breach|tat.*out|sdp delay', na=False),
    df['Clean_Text'].str.contains(r'claim.*reg|register.*claim|raised.*claim|^claim registrations$', na=False),
    df['Clean_Text'].str.contains(r'fake.*clos|closed.*without|cancel.*without|wrongly.*clos', na=False),
    df['Clean_Text'].str.contains(r'fake.*stat|fake.*updat|fake.*info|lying|lied', na=False),
    df['Clean_Text'].str.contains(r'warranty|ew -|extended warranty|amc|expired', na=False),
    df['Clean_Text'].str.contains(r'check.*status|status.*call|follow up|cx.*know', na=False),
    df['Clean_Text'].str.contains(r'delay|tat breach|exceed|slow|waiting', na=False)
]

choices = [
    'Delay in Service - Out of SPD', 
    'Claim Registrations', 
    'Fake closure', 
    'Fake status', 
    'Warranty Related', 
    'To Check Status Of The Call', 
    'Delay in service'
]

df['Category'] = np.select(conditions, choices, default='Uncategorized')

uncat_mask = df['Category'] == 'Uncategorized'
docs = df.loc[uncat_mask, 'Clean_Text'].tolist()

if len(docs) > 0:
    print(f"Running BERTopic on {len(docs)} uncategorized rows...")
    
    topic_model = BERTopic(language="english", min_topic_size=2)
    topics, probs = topic_model.fit_transform(docs)
    
    df.loc[uncat_mask, 'AI_Topic_ID'] = topics
      
    df['Final_Bifurcation'] = np.where(
        df['Category'] == 'Uncategorized', 
        'AI Discovery: ' + df['AI_Topic_ID'].astype(str), 
        df['Category']
    )
else:
    df['Final_Bifurcation'] = df['Category']

print("\n---- DISCOVERED TOPICS ")
if len(docs) > 0:
    print(topic_model.get_topic_info()[['Topic', 'Count', 'Name']].head(10))

print("\n---- FINAL BIFURCATION SUMMARY ----")
summary = df['Final_Bifurcation'].value_counts(normalize=True).reset_index()
summary.columns = ['Category', 'Percentage']
summary['Percentage'] = (summary['Percentage'] * 100).round(2)
print(summary)

output_data = r"C:\Users\rajeshkumar.t\Desktop\converted_opt.xlsx"

df.to_excel(output_data, index=False)
