from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone,PodSpec,ServerlessSpec
import pandas as pd
import json
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm

def insert_pinecone(filepath):

    pc = Pinecone(api_key="49be9ffc-9f65-4cd4-8b98-85e8120eab9b")

    # pc.create_index(
    #     name="piflow",
    #     dimension=384, # Replace with your model dimensions
    #     metric="cosine", # Replace with your model metric
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     ) 
    # )

    # print(pc.describe_index('piflow'))

    with open(filepath,'r') as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    bm25 = BM25Encoder()
    bm25.fit(df['text'])
    # print("BM25 encoder created successfully")

    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i+batch_size, len(df)) # 确定当前批次的结束索引
        df_batch = df.iloc[i:i_end]  # 从 DataFrame 中提取当前批次
        df_dict = df_batch.to_dict(orient="records") # 将批次转换为字典列表

        meta_batch = [
    " ".join(map(str, x)) for x in df_batch.loc[:, ~df_batch.columns.isin(['Filetype', 'Element Type', 'Date Modified'])].values.tolist()]
        text_batch = df['text'][i:i_end].tolist()
        sparse_embeds = bm25.encode_documents([text for text in meta_batch])
        dense_embeds = model.encode(text_batch).tolist()
        ids = [str(x) for x in range(i, i_end)]
        upserts = []

        for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, df_dict):
            upserts.append({'id': _id,'sparse_values': sparse,'values': dense,'metadata': meta})
            try:
                index = pc.Index(host='piflow-2025-0uaxktv.svc.aped-4627-b74a.pinecone.io')
                index.upsert(upserts)
            except Exception as e:
                print(f"An error occurred: {e}")