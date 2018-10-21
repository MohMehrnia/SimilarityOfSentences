# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import numpy as np
from pyravendb.store import document_store
from DocumentObjects import Document
# import pandas as pd

# data = pd.read_excel("dataset.xlsx")
# data = list(data["description"].astype(str).values.flatten())


if __name__=='__main__':
    store = document_store.DocumentStore(urls=["http://localhost:8080"], database="SeSimi")
    store.initialize()
    with store.open_session() as session:
        # [session.store(Document(i, data[i]))  for i in range(0, len(data)) if data[i]!="description"]
        # session.save_changes()
        result = list(session.query(collection_name='Documents'))
        print(result[5].title)

        [print('{0}, {1}'.format(result[i].title, result[i].key))
            for i in range(0, len(result))]
