import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],

       	['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],

       	['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],

       	['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],

      ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]

"""
preaparing the dataser
"""
te=TransactionEncoder()
te_ar=te.fit(dataset).transform(dataset)
df1=pd.DataFrame(te_ar,columns=te.columns_)






"""
support
"""
frequent_itemsets =apriori(df1,min_support=0.6,use_colnames=True)
print(frequent_itemsets)
"""
Confidence 
"""
Confidence=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)
print(Confidence)
"""
lift
"""
lift=association_rules(frequent_itemsets,metric="lift",min_threshold=1)
print(lift)



"""
result:
it more likly to buy Pants and Skirt together than Skirt alone or Pants alone
however it's more likly to buy Pants after buing Skirt.
"""









