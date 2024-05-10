from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone,PodSpec,ServerlessSpec

pc = Pinecone(api_key="49be9ffc-9f65-4cd4-8b98-85e8120eab9b")


pc.create_index(
    name="piflow",
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Create a pod index. By default, Pinecone indexes all metadata.
# pc.create_index( 
#   name='piflow-2025', 
#   dimension=10, 
#   metric='dotproduct', 
#   spec=PodSpec( 
#     environment='gcp-starter', 
#     pod_type='p1.x1', 
#     pods=1, ))

# Show the information about the newly-created index
print(pc.describe_index('piflow'))
  


#当构建RAG 系统时，可以构造一个函数函数删除那些“UncategorizedText”元素。
#我认为这一步应该只是查询的时候需要，在存储的时候不需要删除，所以这一步暂时不考虑




#convert JSON output into Pandas DataFrame
import pandas as pd
import json


with open('/Users/mrniu/Desktop/GitHub/piflow_pinecone/output_ocr_1.json','r') as f:
    data = json.load(f)


df = pd.json_normalize(data)


# print(df)

  
# 定义稀疏向量和稠密向量
# 使用BM25编码器创建的稀疏向量对于基于关键字的数据特别有用。使用 SentenceTransformer 模型生成的密集向量捕获文本的语义本质。

# sparse vector

from pinecone_text.sparse import BM25Encoder

bm25 = BM25Encoder()
bm25.fit(df['text'])
print("BM25 encoder created successfully")
# dense vector

print(5)
# model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
print(7)


from tqdm.auto import tqdm
print(9)
batch_size = 32
# 以 'batch_size' 的大小遍历 DataFrame 'df'
for i in tqdm(range(0, len(df), batch_size)):
    print(8)
    i_end = min(i+batch_size, len(df)) # 确定当前批次的结束索引
    df_batch = df.iloc[i:i_end]  # 从 DataFrame 中提取当前批次
    df_dict = df_batch.to_dict(orient="records") # 将批次转换为字典列表
# 通过连接除 'Filetype', 'Element Type', 和 'Date Modified' 之外的所有列来创建元数据批次
print(0)
meta_batch = [
" ".join(map(str, x)) for x in df_batch.loc[:, ~df_batch.columns.isin(['Filetype', 'Element Type', 'Date Modified'])].values.tolist()]
# 从当前批次中提取 'text' 列作为列表
text_batch = df['text'][i:i_end].tolist()
# 使用 bm25 算法对元数据批次进行编码，创建稀疏嵌入
sparse_embeds = bm25.encode_documents([text for text in meta_batch])
# 使用模型对文本批次进行编码，创建密集嵌入
dense_embeds = model.encode(text_batch).tolist()
# 为当前批次生成 ID 列表
ids = [str(x) for x in range(i, i_end)]
# 初始化一个列表并遍历批次中的每个项目以准备 upsert 数据
upserts = []
print(1)
for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, df_dict):
    upserts.append({'id': _id,'sparse_values': sparse,'values': dense,'metadata': meta})
    # 连接到 Pinecone 索引并 upsert 批次数据
    try:
      print(2)
      index = pc.Index(host='piflow-2025-0uaxktv.svc.aped-4627-b74a.pinecone.io')
      index.upsert(upserts)
    except Exception as e:
      print(f"An error occurred: {e}")