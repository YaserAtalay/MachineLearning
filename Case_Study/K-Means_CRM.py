

########################################
# Kmeans vs Kural Tabanlı Segmentasyon
#########################################

# gerekli importlar

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# veri seti
df = pd.read_excel("/Users/dlaraalcan/Desktop/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df.head()
df.shape


# null degerler
df.isnull().sum()

# Description'da 1454
# Customer ID'de 135080 eksik deger var.

# eksik degerleri siliyoruz.
df.dropna(inplace= True)

# kontrol için tekrar bakalım eksik deger sayısına
df.isnull().sum()

# fatura numarası C ile baslayan iadeleri cıkarıyoruz.
df = df[~df["Invoice"].str.contains("C", na=False)]

# aykırı degerleri belirliyoruz
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# aykırı degerleri limitlere baskılıyoruz.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# quantity ve price değişkenleri için aykırı degerleri baskılıyoruz.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


# fatura başına elde edilen toplam kazanç
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.head()
# RFM metriklerinin hesaplanması:
# analizi hangi tarihe göre yapmak istiyorsak o tarihi belirliyoruz.
today_date = dt.datetime(2011, 12, 11)

df["InvoiceDate"].dtype

# her müşteriye ait degerler
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


# olustutulan metriklerin isimlerini değiştiriyoruz
rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.head()


# monetary degeri 0 dan büyük olacak sekilde filtrelendirme yapıyoruz.
rfm = rfm[rfm['Monetary'] > 0]


# RFM metriklerini skorlara ceviriyourz.
# recency degerinde bir farklılık vardı, en yeni ziyaret edenler daha yüksek skor almalı bu sebeple en kücük recency degerine sahipler en yüksek skoru alır.
rfm['recency_score'] = pd.qcut(rfm['Recency'], 5, [5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, [1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['Monetary'], 5, [1, 2, 3, 4, 5])

# metriklerin skorlarını birleştiriyoruz.
# monetary metrigini hesaba katmıyoruz.
rfm['RFM_SCORE'] = (rfm['recency_score'].astype('str') + rfm['frequency_score'].astype('str'))

rfm.head()

# skor segmentleri
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


# skorları segmentlere göre sınıflıyoruz
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.head()


# K-MEANS
####################################

pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

selected = rfm[['Recency', 'Frequency']]
selected.head()

#  recency ve frequency için standartlastırma yapıyoruz
sc = MinMaxScaler()
sc.fit(selected)


scaled = sc.transform(selected) # scaled =  metriklerin standartlastırılmıs hali.
# yeni bir df olusturuyoruz standartlastırılmıs recency ve frequency degerlerini gösteren:
scaled_df = pd.DataFrame(scaled, columns=selected.columns,index=selected.index)
scaled_df.head()

""""
# optimum küme sayısını elbow yöntemi ile bulmak:

kmeans = KMeans(random_state=17)
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=17).fit(scaled_df)
    ssd.append(kmeans.inertia_)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(1, 30))
elbow.fit(scaled_df)
elbow.show()
elbow.elbow_value_ 

# elbow yöntemi optimum kume sayısını 4 olarak buldu.  """

# Final Cluster'ların Oluşturulması: (bulunan optimum kume sayısına göre)
kmeans = KMeans(n_clusters=4, random_state=17).fit(scaled_df)
kumeler = kmeans.labels_

# kumeleri verisetine ekliyoruz.
rfm["cluster_no"] = kumeler
rfm["cluster_no"] = rfm["cluster_no"] + 1 # 0'dan baslamaması için

rfm.head()

# k-means yöntemi ile olusturulan kumelerde kacar kişi var?
rfm.groupby("cluster_no").agg({"cluster_no": "count"})

rfm.groupby(["cluster_no","segment"]).agg({"segment": "count"})


rfm.groupby("cluster_no").agg({"Recency":["mean","min","max"],
                                  "Frequency":["mean","min","max"]
                       })
