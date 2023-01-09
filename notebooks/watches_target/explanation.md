**dataset_0** = [age_access_type - num
 average_rating - num 
duration - num
 type - cat 
release_year - cat
actor - top 5 cat
country - top 1 cat
genre - top 3 cat director - top 1 cat
availability OHE cat
subscription_only cat]

**dataset_1** = datset_0 + name ruBERT embeddings

**dataset_2** = cat2vec

    **dataset2_1** = cat2vec min_count=1

**dataset_3** = ohe всё

Catalogue + Kinopoisk
**dataset_0kp** = [age_access_type cat
        duration num
        type cat
        actor, country, director, producer, writer - top n in list
        genre tf-df
        budget num
        marketing num
        rus bo num
        words bo num
        ]

**dataset_2kp** = cat2vec
