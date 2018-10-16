from __future__ import unicode_literals
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Normalizer, word_tokenize, Lemmatizer


if __name__=="__main__":
    data = ["من به یادگیری ماشین بسیار علاقه مند هستم",
            "من عاش کد نویسی با پایتون هستم",
            "من عاشق ساختن نرم افزارهای هوشمند هستم",
            "هوشمند سازی یک نرم افزار فرآیندی بسیار پیچیده است"]        

    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    tagged_data = [TaggedDocument(words=word_tokenize(lemmatizer.lemmatize(normalizer.normalize(_d.lower()))), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vec_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1, workers=4)
    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    test_data = word_tokenize("من برنامه نویس پایتون هستم".lower())
    result = model.docvecs.most_similar([model.infer_vector(test_data)], topn=5)
    print(result)
    

