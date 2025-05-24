import gc
from fastapi import FastAPI
from .schemas import (
    StringUpsertRequest,
    StringListUpsertRequest,
    DeleteDocWithIdRequest,
    DeleteDocWithTitleRequest,
    DeleteChunkRequest,
    DeleteUserCollectionRequest,
    UpdateRequest,
    QueryRequest,
    QueryOnDocRequest,
    ScrollRequest,
)
from ..core.document_ingestor import DocumentProcessor
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ..core.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_HOST,
    EMBEDDING_PORT,
)


data_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    This function initializes and tears down the global `data_processor` used 
    throughout the application for handling document processing. It is designed 
    to be used with FastAPI's `lifespan` parameter to manage application-level 
    startup and shutdown events.

    On startup:
        - Instantiates `DocumentProcessor` using configuration values from environment.
        - Assigns the instance to a global variable `data_processor`.

    On shutdown:
        - Deletes the global `data_processor` instance.
        - Explicitly sets it to None and triggers garbage collection.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """
    global data_processor
    data_processor = DocumentProcessor(
        qdrant_host = QDRANT_HOST,
        qdrant_port = QDRANT_PORT,
        embedding_host = EMBEDDING_HOST,
        embedding_port = EMBEDDING_PORT,
    )

    yield
    del data_processor
    data_processor = None
    gc.collect()


app = FastAPI(
    title = "Vector DB Handler",
    lifespan = lifespan,
)


@app.get(
    path = "/",
    tags = [
        "Upsert Processor",
        "Delete Processor",
        "Query Processor",
        "List Processor",
    ],
)
async def root():
    """
    Root endpoint for health check.

    Returns a simple JSON message indicating that the service is running.
    This endpoint is commonly used to verify that the FastAPI app is live and reachable.

    Returns:
        JSONResponse: A JSON object with a health check message.
    """
    return JSONResponse(
        {
            "message": "Service is up and running",
        }
    )


@app.post(
    path = "/upsert_data/",
    tags = [
        "Upsert Processor",
    ],
)
async def upsert_data(
    request: StringUpsertRequest
):
    """
    Upserts a string chunk along with its metadata into the vector database.

    This endpoint takes in a user ID, a chunk of text, and metadata. The text is 
    embedded and stored in the Qdrant vector database with the associated metadata.

    Args:
        request (StringUpsertRequest): The request body containing:
            - user_id (str): Unique identifier for the user.
            - chunk (str): The text content to embed and store.
            - metadata (dict): Information related to the chunk (e.g., DocId, ChunkId, Title).

    Returns:
        JSONResponse: A success message containing the user ID and metadata of the upserted data.
    """
    user_id = request.user_id
    chunk = request.chunk
    metadata = request.metadata
    data_processor.upsert_string(
        user_id = user_id,
        chunk = chunk,
        metadata = metadata,
    )
    return JSONResponse(
        {
            "Message": "String data was successfully upserted to the vector database.",
            "User-Id": user_id,
            "Upserted-Metadata": metadata,
        }
    )


@app.post(
    path = "/upsert_list_data/",
    tags = [
        "Upsert Processor",
    ],
)
async def upsert_list_data(
    request: StringListUpsertRequest
):
    """
    Upserts a list of string chunks and their corresponding metadata into the vector database.

    This endpoint accepts multiple chunks of text along with associated metadata for each chunk,
    embeds them, and stores them in the vector database under the specified user ID.

    Args:
        request (StringListUpsertRequest): The request body containing:
            - user_id (str): Unique identifier for the user.
            - chunks (List[str]): A list of text strings to embed and store.
            - metadatas (List[dict]): A list of metadata dictionaries corresponding to each chunk.
              Each dictionary typically includes identifiers like DocId, ChunkId, and Title.

    Returns:
        JSONResponse: A success message containing the user ID and the list of upserted metadata entries.
    """
    user_id = request.user_id
    chunks = request.chunks
    metadatas = request.metadatas
    data_processor.upsert_str_list(
        user_id = user_id,
        chunks = chunks,
        metadatas = metadatas,
    )
    return JSONResponse(
        {
            "Message": "List of strings data were successfully upserted to the vector database.",
            "User-Id": user_id,
            "Upserted-Metadatas": metadatas,
        }
    )


@app.delete(
    path = "/delete_doc/",
    tags = [
        "Delete Processor",
    ],
)
async def delete_doc(
    request: DeleteDocWithIdRequest
):
    """
    Deletes all vectors associated with the specified `DocId` for a user.

    Args:
        request (DeleteDocWithIdRequest): A JSON body containing:
            - user_id (str): The user's ID
            - doc_id (int): The document ID to delete

    Returns:
        JSONResponse: Confirmation of successful deletion.
    """
    user_id = request.user_id
    doc_id = request.doc_id
    data_processor.delete_doc(
        user_id = user_id,
        doc_id = doc_id,
    )
    return JSONResponse(
        {
            "Message": f"All documents with DocId={doc_id} were successfully deleted.",
            "User-Id": user_id
        }
    )


@app.delete(
    path = "/delete_doc_by_title/",
    tags = [
        "Delete Processor",
    ],
)
async def delete_doc_by_title(
    request: DeleteDocWithTitleRequest
):
    """
    Deletes all documents for a given user that match the specified title.

    Args:
        request (DeleteDocWithTitleRequest): A JSON body containing:
            - user_id (str): The ID of the user
            - doc_title (str): The title of the document(s) to delete

    Returns:
        JSONResponse: Confirmation message.
    """
    user_id = request.user_id
    doc_title = request.doc_title
    data_processor.delete_doc_by_title(
        user_id = user_id,
        doc_title = doc_title,
    )
    return JSONResponse(
        {
            "Message": f"All documents with Title = \"{doc_title}\" were successfully deleted.",
            "User-Id": user_id
        }
    )


@app.delete(
    path = "/delete_chunk/",
    tags = [
        "Delete Processor",
    ],
)
async def delete_chunk(
    request: DeleteChunkRequest
):
    """
    Deletes a specific chunk of a document for a given user.

    Args:
        request (DeleteChunkRequest): A JSON body containing:
            - user_id (str): The ID of the user.
            - doc_id (int): The document ID.
            - chunk_id (int): The chunk ID to be deleted.

    Returns:
        JSONResponse: Confirmation of successful deletion.
    """
    user_id = request.user_id
    doc_id = request.doc_id
    chunk_id = request.chunk_id
    data_processor.delete_chunk(
        user_id = user_id,
        doc_id = doc_id,
        chunk_id = chunk_id,
    )
    return JSONResponse(
        {
            "Message": f"Document with DocId={doc_id} and ChunkId={chunk_id} was successfully deleted.",
            "User-Id": user_id
        }
    )


@app.delete(
    path = "/delete_user_collection_data/",
    tags = [
        "Delete Processor",
    ],
)
async def delete_user_collection_data(
    request: DeleteUserCollectionRequest
):
    """
    Deletes the entire vector collection data associated with a specific user.

    This operation is irreversible and removes all stored embeddings and metadata
    related to the user's collection from the vector database.

    Args:
        request (DeleteUserCollectionRequest): A JSON body containing:
            - user_id (str): The ID of the user whose collection should be deleted.

    Returns:
        JSONResponse: Confirmation of successful deletion.
    """
    user_id = request.user_id
    data_processor.delete_user_collection_data(
        user_id = user_id,
    )
    return JSONResponse(
        {
            "Message": f"All data in the collection for user '{user_id}' were successfully deleted.",
        }
    )


@app.delete(
    path = "/delete_user_collection/",
    tags = [
        "Delete Processor",
    ],
)
async def delete_user_collection(
    request: DeleteUserCollectionRequest
):
    """
    Deletes the entire vector collection for the specified user.

    This operation is irreversible and removes user's collection from the vector database.

    Args:
        request (DeleteUserCollectionRequest): Request body containing:
            - user_id (str): Unique identifier for the user whose collection will be deleted.

    Returns:
        JSONResponse: A confirmation message with the user ID.
    """
    user_id = request.user_id
    data_processor.delete_user_collection(
        user_id = user_id,
    )
    return JSONResponse(
        {
            "Message": f"User collection for user_id = '{user_id}' was successfully deleted.",
        }
    )


@app.put(
    path = "/update_data/",
    tags = [
        "Upsert Processor",
    ],
)
async def update_data(
    request: UpdateRequest
):
    """
    Updates the vector embedding of a specific chunk of text for a given user.

    This endpoint receives a new string ('chunk') and updates the embedding for the
    chunk identified by 'doc_id' and 'chunk_id'. Metadata remains unchanged.

    Args:
        request (UpdateRequest): Request body containing:
            - user_id (str): Unique identifier of the user.
            - chunk (str): The new string content to embed and update.
            - doc_id (int): Document ID to identify the document.
            - chunk_id (int): Chunk ID to identify the specific chunk within the document.

    Returns:
        JSONResponse: Confirms the successful update of the chunk's vector embedding,
        including the user ID, document ID, and chunk ID.
    """
    user_id = request.user_id
    chunk = request.chunk
    doc_id = request.doc_id
    chunk_id = request.chunk_id
    data_processor.update_chunk(
        user_id = user_id,
        chunk = chunk,
        doc_id = doc_id,
        chunk_id = chunk_id,
    )
    return JSONResponse(
        {
            "Message": f"String data for DocId={doc_id} ChunkId={chunk_id} was successfully updated.",
            "User-Id": user_id,
        }
    )


@app.post(
    path = "/search_query/",
    tags = [
        "Query Processor",
    ],
)
async def search_query(
    request: QueryRequest
):
    """
    Search for vectors in a user's collection based on a text query.

    Args:
        request (QueryRequest): 
            - user_id (str): Unique identifier for the user whose collection to search.
            - query (str): The text string to vectorize and search; may be arbitrarily long.
            - limit (int): Maximum number of results to return.

    Returns:
        JSONResponse: A JSON object with a "Results" key containing a list of matched items, each including:
            - DocId (int)
            - ChunkId (int)
            - Title (str)
            - Similarity Score (float)
    """
    user_id = request.user_id
    query = request.query
    limit = request.limit
    results = data_processor.search_query(
        user_id = user_id,
        string_query = query,
        limit = limit,
    )
    return JSONResponse(
        {
            "Results": results
        }
    )


@app.post(
    path = "/search_query_on_doc/",
    tags = [
        "Query Processor",
    ],
)
async def search_query_on_doc(
    request: QueryOnDocRequest
):
    """
    Search for vectors within specific documents in a user's collection based on a text query.

    Args:
        request (QueryOnDocRequest):
            - user_id (str): Unique identifier for the user whose collection to search.
            - query (str): The text string to vectorize and search within the specified documents.
            - doc_ids (List[int]): List of document IDs to restrict the search scope.
            - limit (int): Maximum number of results to return.

    Returns:
        JSONResponse: A JSON object with a "Results" key containing a list of matched items within
        the specified documents, each including:
            - DocId (int)
            - ChunkId (int)
            - Title (str)
            - Similarity Score (float)
    """
    user_id = request.user_id
    query = request.query
    doc_ids = request.doc_ids
    limit = request.limit
    results = data_processor.search_query_on_doc(
        user_id = user_id,
        doc_ids = doc_ids,
        string_query = query,
        limit = limit,
    )
    return JSONResponse(
        {
            "Results": results
        }
    )


@app.post(
    path = "/scroll_user_collection/",
    tags = [
        "List Processor",
    ],
)
async def scroll_user_collection(
    request: ScrollRequest
):
    """
    Retrieve a limited list of vector data chunks from the specified user's collection.

    Args:
        user_id (str): Unique identifier of the user.
        limit (int, optional): Maximum number of records to return. Defaults to 20.

    Returns:
        JSONResponse: A JSON object containing a list of chunked metadata and content up to the specified limit.
    """
    user_id = request.user_id
    limit = request.limit
    results = data_processor.scroll_user_collection(
        user_id = user_id,
        limit = limit
    )
    return JSONResponse(
        {
            f"User {user_id} data": results
        }
    )

@app.get(
    path = "/list_users_collection/",
    tags = [
        "List Processor",
    ],
)
async def list_users_collection():
    """
    Retrieve the names of all existing user collections in the vector database.

    Returns:
        JSONResponse: A JSON object containing a list of collection names.
    """
    results = data_processor.list_collections()
    return JSONResponse(
        {
            f"User-Collections": results
        }
    )
