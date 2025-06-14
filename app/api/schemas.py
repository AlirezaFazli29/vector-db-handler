from pydantic import BaseModel
from typing import List, Dict, Any

class StringUpsertRequest(BaseModel):
    user_id: str
    chunk: str
    metadata: Dict[str, Any] = {
        "DocId": 0,
        "ChunkId": 0,
        "Title": "string"
    }

class StringListUpsertRequest(BaseModel):
    user_id: str
    chunks: List[str]
    metadatas: List[Dict[str, Any]] = [
        {
            "DocId": 0,
            "ChunkId": 0,
            "Title": "string"
        }
    ]

class DeleteDocWithIdRequest(BaseModel):
    user_id: str
    doc_id: int

class DeleteDocWithTitleRequest(BaseModel):
    user_id: str
    doc_title: str

class DeleteChunkRequest(BaseModel):
    user_id: str
    doc_id: int
    chunk_id: int

class DeleteByIdRequest(BaseModel):
    user_id: str
    vector_id: str

class DeleteListByIdRequest(BaseModel):
    user_id: str
    vector_ids: List[str]

class DeleteUserCollectionRequest(BaseModel):
    user_id: str

class UpdateRequest(BaseModel):
    user_id: str
    chunk: str
    doc_id: int
    chunk_id: int

class QueryRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 5

class QueryOnDocRequest(BaseModel):
    user_id: str
    query: str
    doc_ids: List[int]
    limit: int = 5

class ScrollDocRequest(BaseModel):
    user_id: str
    doc_id: int
    limit: int = 20

class ScrollChunkRequest(BaseModel):
    user_id: str
    doc_id: int
    chunk_id: int
    limit: int = 20

class ScrollDocsRequest(BaseModel):
    user_id: str
    doc_ids: List[int]
    limit: int = 20

class ScrollRequest(BaseModel):
    user_id: str
    limit: int = 20