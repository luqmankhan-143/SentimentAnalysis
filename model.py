   
def prediction(name,reviews):
    print(name)
    pipeline = pickle.load(open('pickle/user_based_recomm.pkl', 'rb'))
    sr = pipeline.loc[name].sort_values(ascending=False)[0:20] ## series
    top_20_products = pd.DataFrame({'name':sr.index})
    top_20_reviews = reviews[reviews['name'].isin(top_20_products['name'])][['name','reviews_text']] 
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open('pickle/vector', 'rb')))
    test_data_features = transformer.fit_transform(loaded_vec.fit_transform(top_20_reviews['reviews_text']))
    loaded_model = pickle.load(open("pickle/LRModel", 'rb'))
    result1 = loaded_model.predict(test_data_features)
    top_20_reviews['sentiment'] = result1.tolist()
    top = top_20_reviews.groupby(['name']).mean()
    top5 = top.sort_values(by='sentiment',ascending=False)[:5]
    top5.reset_index(level=0, inplace=True)
    top_5_products= top5['name']
    top_5_product = pd.DataFrame({'name':top_5_products})
    return top_5_product  


