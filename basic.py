from __future__ import unicode_literals

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Lemmatizer, Normalizer, word_tokenize

if __name__ == "__main__":
    data = ["من به یادگیری ماشین بسیار علاقه‌مند هستم",
            "من عاشق کد‌نویسی با پایتون هستم",
            "من عاشق ساختن نرم‌افزارهای هوشمند هستم",
            "هوشمند‌سازی یک نرم‌افزار فرآیندی بسیار پیچیده است"]

    data = [normalizer.normalize(_d) for i, _d in enumerate(data)]
    lemmatizer = Lemmatizer()
    tagged_data = [TaggedDocument(
        words=[lemmatizer.lemmatize(_d) for i, _d in enumerate(
            word_tokenize(normalizer.normalize(_d.lower())))],
        tags=[str(i)]) for i, _d in enumerate(data)]

    print(tagged_data)
    max_epochs = 100
    vec_size = 40
    alpha = 0.025

    model = Doc2Vec(vec_size=vec_size, alpha=alpha,
                    min_alpha=0.00025, min_count=1, dm=1, workers=4)
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("BasicModel")
    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)
