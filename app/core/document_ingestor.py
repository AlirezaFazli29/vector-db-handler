import json
import uuid
import logging
import requests
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    VectorParams,
    PointStruct,
    MatchValue,
    Distance,
    Filter,
)
from qdrant_client.http.models import (
    ScoredPoint,
    Record,
)


logger = logging.getLogger(__name__)


class QdrantHandler:
    """
    A handler for managing user-specific collections and vector data within a Qdrant vector database.

    Each user has a dedicated collection named "user_<user_id>", where hyphens in the user_id are 
    replaced with underscores.
    Provides methods to create collections, insert vectors, delete documents or chunks, search vectors, 
    and scroll collections.

    Args:
        qdrant_host (str): The host address of the Qdrant server.
        qdrant_port (int): The port number of the Qdrant server.
    """
    def __init__(
            self,
            qdrant_host: str,
            qdrant_port: int,
    ):
        """
        Initialize the Qdrant client connection.

        Args:
            qdrant_host (str): Hostname or IP address of the Qdrant service.
            qdrant_port (int): Port number where Qdrant is running.
        """
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
        )
        
    def __collection_name(
            self,
            user_id: str,
    ) -> str:
        """
        Generate the collection name for a given user.

        Args:
            user_id (str): The user's unique identifier.

        Returns:
            str: Collection name in the format 'user_<user_id>' with hyphens replaced by underscores.
        """
        return f"user_{user_id.replace('-', '_')}"
    
    def ensure_user_collection(
            self,
            user_id: str,
            vector_size: int = 1024,
    ) -> None:
        """
        Ensure a collection exists for the user; create it if missing.

        Args:
            user_id (str): User ID to create the collection for.
            vector_size (int, optional): Dimensionality of the vectors. Defaults to 1024.
        """
        collection_name = self.__collection_name(user_id)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def upsert_vector(
            self,
            user_id: str,
            vector: List,
            metadata: Dict,
    ) -> None:
        """
        Insert a vector and its metadata into the user's collection.

        Args:
            user_id (str): The user's unique identifier.
            vector (List): The vector embedding to upsert.
            metadata (Dict): Payload data associated with the vector.
        """
        collection_name = self.__collection_name(user_id)
        point = PointStruct(
            id = str(uuid.uuid4()),
            vector = vector,
            payload = metadata,
        )
        self.client.upsert(
            collection_name = collection_name,
            points = [point]
        )

    def upsert_list_of_vectors(
            self,
            user_id: str,
            vectors: List[List],
            metadatas: List[Dict],
    ) -> None:
        """
        Bulk-upsert vectors (and their payloads) into the user's collection.

        Args:
            user_id (str): The user's unique identifier.

            vectors (List[List]): List of vector embeddings to store.  

            metadatas (List[Dict]): List of payload dictionaries that accompany each vector

        Raises:
            ValueError: If the lengths of ``vectors`` and ``metadatas`` differ.
        """
        collection_name = self.__collection_name(user_id)
        points = [
            PointStruct(
                id = str(uuid.uuid4()),
                vector = vectors[i],
                payload = metadatas[i]
            ) for i in range(len(vectors))
        ]
        self.client.upsert(
            collection_name = collection_name,
            points = points,
        )

    def delete_doc(
            self,
            user_id: str,
            doc_id: int,
    ) -> None:
        """
        Delete all vectors associated with a specific document ID in the user's collection.

        Args:
            user_id (str): The user's unique identifier.
            doc_id (int): Document ID to delete.
        """
        collection_name = self.__collection_name(user_id)
        filter_condition = Filter(
            must = [
                FieldCondition(key="DocId", match=MatchValue(value=doc_id))
            ]
        )
        self.client.delete(
            collection_name = collection_name,
            points_selector = filter_condition,
        )

    def delete_doc_by_title(
            self,
            user_id: str,
            doc_title: str,
    ) -> None:
        """
        Delete all vectors matching a document title in the user's collection.

        Args:
            user_id (str): The user's unique identifier.
            doc_title (str): Title of the document to delete.
        """
        collection_name = self.__collection_name(user_id)
        filter_condition = Filter(
            must = [
                FieldCondition(key="Title", match=MatchValue(value=doc_title))
            ]
        )
        self.client.delete(
            collection_name = collection_name,
            points_selector = filter_condition,
        )

    def delete_chunk(
            self,
            user_id: str,
            doc_id: int,
            chunk_id: int,
    ) -> None:
        """
        Delete a specific chunk of a document from the user's collection.

        Args:
            user_id (str): The user's unique identifier.
            doc_id (int): Document ID of the chunk.
            chunk_id (int): Chunk ID within the document.
        """
        collection_name = self.__collection_name(user_id)
        filter_condition = Filter(
            must=[
                FieldCondition(key="DocId",  match=MatchValue(value=doc_id)),
                FieldCondition(key="ChunkId", match=MatchValue(value=chunk_id)),
            ]
        )
        self.client.delete(
            collection_name = collection_name,
            points_selector = filter_condition,
        )

    def delete_user_collection_data(
            self,
            user_id,
    ) -> None:
        """
        Delete all vectors from the user's collection but keep the collection itself.

        Args:
            user_id (str): The user's unique identifier.
        """
        collection_name = self.__collection_name(user_id)
        filter_condition = Filter(must=[])
        self.client.delete(
            collection_name = collection_name,
            points_selector = filter_condition,
        )

    def delete_user_collection(
            self,
            user_id: str,
    ) -> None:
        """
        Delete the entire user collection including all data and metadata.

        Args:
            user_id (str): The user's unique identifier.
        """
        collection_name = self.__collection_name(user_id)
        self.client.delete_collection(collection_name)

    def update_vector(
            self,
            user_id: str,
            vector: List,
            doc_id: int,
            chunk_id: int,
    ) -> None:
        """
        Update the vector of a data point identified by document ID and chunk ID in the user's collection.

        Args:
            user_id (str): The user identifier to determine the collection name.
            vector (List): The new vector embedding to update.
            doc_id (int): The document ID of the data point.
            chunk_id (int): The chunk ID within the document to identify the exact data point.
        """
        collection_name = self.__collection_name(user_id)
        filter_condition = Filter(
            must=[
                FieldCondition(key="DocId",  match=MatchValue(value=doc_id)),
                FieldCondition(key="ChunkId", match=MatchValue(value=chunk_id)),
            ]
        )
        response = self.client.scroll(
            collection_name = collection_name,
            scroll_filter = filter_condition,
            limit = 1,
            with_payload = True,
            with_vectors = False,
        )
        if response[0]:
            record = response[0][0]
            point = PointStruct(
                id = record.id,
                payload = record.payload,
                vector = vector
            )
            self.client.upsert(
                collection_name = collection_name,
                points = [point]
            )

    def search_query(
            self,
            user_id: str,
            vector_query: List,
            limit: int = 5,
            with_payload: bool = True,
            with_vectors: bool = False,
            score_threshold: float = 0,
    ) -> List[ScoredPoint]:
        """
        Search for vectors nearest to the query vector in the user's collection.

        Args:
            user_id (str): The user's unique identifier.
            vector_query (List): Query vector for similarity search.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
            with_payload (bool, optional): Include payload in results. Defaults to True.
            with_vectors (bool, optional): Include vector data in results. Defaults to False.
            score_threshold (float, optional): Minimum similarity score threshold. Defaults to 0.

        Returns:
            List[ScoredPoint]: List of scored points matching the query.
        """
        collection_name = self.__collection_name(user_id)
        response = self.client.query_points(
            collection_name = collection_name,
            query = vector_query,
            limit = limit,
            with_payload = with_payload,
            with_vectors = with_vectors,
            score_threshold = score_threshold,
        )
        return response.points

    def search_query_on_doc(
            self,
            user_id: str,
            doc_ids: List[int],
            vector_query: List,
            limit: int = 5,
            with_payload: bool = True,
            with_vectors: bool = False,
            score_threshold: float = 0,
    ) -> List[ScoredPoint]:
        """
        Search vectors nearest to the query vector limited to specified document IDs.

        Args:
            user_id (str): The user's unique identifier.
            doc_ids (List[int]): List of document IDs to filter search.
            vector_query (List): Query vector for similarity search.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
            with_payload (bool, optional): Include payload in results. Defaults to True.
            with_vectors (bool, optional): Include vector data in results. Defaults to False.
            score_threshold (float, optional): Minimum similarity score threshold. Defaults to 0.

        Returns:
            List[ScoredPoint]: List of scored points matching the query and document filter.
        """
        collection_name = self.__collection_name(user_id)
        f = [
            FieldCondition(
                key = "DocId",
                match = MatchValue(value=i)
            ) for i in doc_ids
        ]
        filter_condition = Filter(should=f)
        response = self.client.query_points(
            collection_name = collection_name,
            query = vector_query,
            limit = limit,
            with_payload = with_payload,
            with_vectors = with_vectors,
            score_threshold = score_threshold,
            query_filter = filter_condition,
        )
        return response.points

    def scroll_collection(
            self,
            user_id: str,
            limit: int,
            with_payload: bool = True,
            with_vectors: bool = False,
    ) -> List[Record]:
        """
        Scroll through the user's collection and retrieve a batch of records.

        Args:
            user_id (str): The user's unique identifier.
            limit (int): Number of records to retrieve.
            with_payload (bool, optional): Include payload in results. Defaults to True.
            with_vectors (bool, optional): Include vector data in results. Defaults to False.

        Returns:
            List[Record]: List of records from the collection.
        """
        collection_name = self.__collection_name(user_id)
        response = self.client.scroll(
            collection_name = collection_name,
            limit = limit,
            with_payload = with_payload,
            with_vectors = with_vectors,
        )
        return response[0]
    
    def list_collections(self) -> list[str]:
        """
        Get the names of every collection that currently exists in Qdrant.

        Returns:
            list[str]: A list of collection names.
        """
        return [
            c.name for c in self.client.get_collections().collections
        ]
    

class DocumentProcessor:
    """
    A processor that integrates with an embedding API and Qdrant vector database to handle text-to-vector
    embedding and storage/retrieval operations.

    This class is responsible for:
      - Communicating with an external embedding service (e.g., to vectorize text strings or lists).
      - Interacting with Qdrant via QdrantHandler to manage vector collections.
      - Using a persistent HTTP session for performance optimization when calling the embedding service.

    Args:
        qdrant_host (str): Hostname or IP address of the Qdrant server.
        qdrant_port (int): Port number where the Qdrant service is running.
        embedding_host (str): Hostname or IP address of the embedding service.
        embedding_port (int): Port number of the embedding service.
    """
    def __init__(
            self,
            qdrant_host: str,
            qdrant_port: int,
            embedding_host: str,
            embedding_port: int,
    ):
        """
        Initializes the processor by setting up the embedding service addresses,
        initializing the Qdrant handler, and preparing a persistent HTTP session with
        appropriate headers for JSON API interaction.

        Args:
            qdrant_host (str): Qdrant server host.
            qdrant_port (int): Qdrant server port.
            embedding_host (str): Embedding API server host.
            embedding_port (int): Embedding API server port.
        """
        embedding_address = f"http://{embedding_host}:{embedding_port}"
        self.embed_str_address = f"{embedding_address}/vectorizer/string/"
        self.embed_list_address = f"{embedding_address}/vectorizer/list/"
        self.qdrant_handler = QdrantHandler(
            qdrant_host = qdrant_host,
            qdrant_port = qdrant_port
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )

    def upsert_string(
            self,
            user_id: str,
            chunk: str,
            metadata: Dict,
    ) -> None:
        """
        Vectorizes a single string using the embedding service and upserts the resulting vector into Qdrant.

        This method:
          - Sends a POST request to the embedding service's string endpoint to obtain a vector representation
            of the input text (chunk).
          - Ensures a vector collection exists for the user (creates it if it doesn't).
          - Upserts the vector and its associated metadata into the user's Qdrant collection.

        Retries up to 5 times on failure before raising an error.

        Args:
            user_id (str): Unique identifier for the user; used to route data to the correct Qdrant collection.
            chunk (str): A string of text to be vectorized.
            metadata (Dict): Metadata associated with the chunk, to be stored with the vector.

        Raises:
            ValueError: If the embedding service fails to return a successful response after 5 attempts.
        """
        payload = {
            "text": chunk,
        }
        for _ in range(5):
            embedding_response = self.session.post(
                url = self.embed_str_address,
                json = payload,
                timeout = 10,
            )
            if embedding_response.status_code == 200:
                response_json = embedding_response.json()
                vector = json.loads(response_json["vectorized text"])
                self.qdrant_handler.ensure_user_collection(user_id)
                self.qdrant_handler.upsert_vector(
                    user_id = user_id,
                    vector = vector,
                    metadata = metadata,
                )
                logger.info(
                    msg = f"Upsert string successful for user_id={user_id}",
                    metadata = metadata
                )
                break
        else:
            error_msg = embedding_response.text
            logger.error(
                msg = f"Failed to vectorize input string for user_id={user_id}: \n{error_msg}"
            )
            raise ValueError(f"Failed to vectorize input string: \n\n{error_msg}")

    def upsert_str_list(
            self,
            user_id: str,
            chunks: List[str],
            metadatas: List[Dict],
    ) -> None:
        """
        Vectorizes a list of strings and upserts the resulting vectors into Qdrant.

        This method performs the following steps:
          - Validates that the number of input text chunks matches the number of metadata entries.
          - Sends a request to the embedding service's list endpoint to obtain vector embeddings.
          - Ensures that a Qdrant collection exists for the specified user.
          - Upserts each vector along with its corresponding metadata into the user's Qdrant collection.

        Retries the embedding request up to 5 times on failure before raising an error.

        Args:
            user_id (str): Unique identifier for the user; used to route data to the correct Qdrant collection.
            chunks (List[str]): List of text strings to be vectorized.
            metadatas (List[Dict]): List of metadata dictionaries, one for each input string.

        Raises:
            ValueError: If the number of chunks and metadata entries do not match.
            ValueError: If the embedding service fails to return a successful response after 5 attempts.
        """
        if len(chunks) != len(metadatas):
            raise ValueError("Length of chunks and metadatas must match.")
        payload = {
            "texts": chunks,
        }
        for _ in range(5):
            embedding_response = self.session.post(
                url = self.embed_list_address,
                json = payload,
                timeout = 10,
            )
            if embedding_response.status_code == 200:
                response_json = embedding_response.json()
                vectors = json.loads(response_json["vectorized texts"])
                self.qdrant_handler.ensure_user_collection(user_id)
                self.qdrant_handler.upsert_list_of_vectors(
                    user_id = user_id,
                    vectors = vectors,
                    metadatas = metadatas,
                )
                logger.info(
                    msg = f"Upsert list of strings successful for user_id={user_id}",
                    metadata = metadatas
                )
                break
        else:
            error_msg = embedding_response.text
            logger.error(
                msg = f"Failed to vectorize input strings for user_id={user_id}: \n{error_msg}"
            )
            raise ValueError(f"Failed to vectorize input strings: \n\n{error_msg}")

    def delete_doc(
            self,
            user_id: str,
            doc_id: int,
    ) -> None:
        """
        Delete all vector data associated with a specific document for a given user.

        This method delegates the deletion to the underlying Qdrant handler and removes
        all chunks stored under the specified document ID for the user.

        Args:
            user_id (str): The unique identifier for the user.
            doc_id (int): The ID of the document to delete.
        """
        self.qdrant_handler.delete_doc(
            user_id = user_id,
            doc_id = doc_id,
        )
    
    def delete_doc_by_title(
            self,
            user_id: str,
            doc_title: str,
    ) -> None:
        """
        Delete all vector data associated with a specific document title for a given user.

        This method delegates the deletion to the underlying Qdrant handler and removes
        all chunks stored under the specified document title for the user.

        Args:
            user_id (str): The unique identifier for the user.
            doc_title (str): The title of the document to delete.
        """
        self.qdrant_handler.delete_doc_by_title(
            user_id = user_id,
            doc_title = doc_title,
        )

    def delete_chunk(
            self,
            user_id: str,
            doc_id: int,
            chunk_id: int,
    ) -> None:
        """
        Delete a specific chunk of vector data for a given user and document.

        This method delegates the deletion to the underlying Qdrant handler
        and removes a single chunk identified by its chunk ID.

        Args:
            user_id (str): The unique identifier for the user.
            doc_id (int): The ID of the document containing the chunk.
            chunk_id (int): The ID of the chunk to delete.
        """
        self.qdrant_handler.delete_chunk(
            user_id = user_id,
            doc_id = doc_id,
            chunk_id = chunk_id,
        )

    def delete_user_collection_data(
            self,
            user_id,
    ) -> None:
        """
        Delete all vector data associated with a specific user.

        This method delegates the deletion of the user's collection to the Qdrant handler.

        Args:
            user_id (str): The unique identifier for the user.
        """
        self.qdrant_handler.delete_user_collection_data(user_id=user_id)

    def delete_user_collection(
            self,
            user_id: str,
    ) -> None:
        """
        Delete the entire collection for a given user from Qdrant, including all data and metadata.

        This permanently removes the user's collection and cannot be undone.

        Args:
            user_id (str): The unique identifier for the user.
        """
        self.qdrant_handler.delete_user_collection(user_id=user_id)

    def update_chunk(
            self,
            user_id: str,
            chunk: str,
            doc_id: int,
            chunk_id: int,
    ) -> None:
        """
        Update an existing vector in Qdrant by re-vectorizing a new chunk of text.

        This method sends the updated chunk to the embedding service, obtains a new vector,
        and replaces the existing vector in Qdrant based on the given document and chunk IDs.

        Args:
            user_id (str): Unique user identifier.
            chunk (str): The updated text to re-vectorize.
            doc_id (int): ID of the document containing the chunk.
            chunk_id (int): ID of the chunk to update within the document.

        Raises:
            ValueError: If vectorization fails after 5 attempts.
        """
        payload = {
            "text": chunk,
        }
        for _ in range(5):
            embedding_response = self.session.post(
                url = self.embed_str_address,
                json = payload,
                timeout = 10,
            )
            if embedding_response.status_code == 200:
                response_json = embedding_response.json()
                vector = json.loads(response_json["vectorized text"])
                self.qdrant_handler.update_vector(
                    user_id = user_id,
                    vector = vector,
                    doc_id = doc_id,
                    chunk_id = chunk_id,
                )
                break
        else:
            error_msg = embedding_response.text
            raise ValueError(f"Failed to vectorize input string: \n\n{error_msg}")
        
    def search_query(
            self,
            user_id: str,
            string_query: str,
            limit: int = 5,
            score_threshold: float = 0,
    ) -> List[Dict[str, Any]]:
        """
        Vectorizes a query string and searches for similar vectors in Qdrant.

        This method sends the query to the embedding service, retrieves the vector,
        and performs a similarity search in Qdrant using that vector.

        Args:
            user_id (str): Identifier for the user's Qdrant collection.
            string_query (str): Text input to vectorize and search with.
            limit (int): Max number of results to return (default 5).
            score_threshold (float): Minimum similarity score for results (default 0).

        Returns:
            List[Dict]: A list of matched chunks with document and score metadata.
        
        Raises:
            ValueError: If the embedding service fails after 5 retries.
        """
        payload = {
            "text": string_query,
        }
        for _ in range(5):
            embedding_response = self.session.post(
                url = self.embed_str_address,
                json = payload,
                timeout = 10,
            )
            if embedding_response.status_code == 200:
                response_json = embedding_response.json()
                vector = json.loads(response_json["vectorized text"])
                query_response = self.qdrant_handler.search_query(
                    user_id = user_id,
                    vector_query = vector,
                    limit = limit,
                    score_threshold = score_threshold,
                )
                logger.info(
                    f"Search query successful for user_id={user_id}, results={len(query_response)}"
                )
                break
        else:
            error_msg = embedding_response.text
            logger.error(
                f"Failed to vectorize input query for user_id={user_id}:\n{error_msg}"
            )
            raise ValueError(f"Failed to vectorize input query: \n\n{error_msg}")
        return [
            {
                "DocId": r.payload.get("DocId"),
                "ChunkId": r.payload.get("ChunkId"),
                "Title": r.payload.get("Title"),
                "Similarity Score": r.score,
            } for r in query_response
        ]
    
    def search_query_on_doc(
            self,
            user_id: str,
            doc_ids: List[int],
            string_query: str,
            limit: int = 5,
            score_threshold: float = 0,
    ) -> List[Dict[str, Any]]:
        """
        Vectorizes a query string and performs a similarity search within specific documents.

        This method sends the query to the embedding service, retrieves the vector, and
        performs a similarity search in Qdrant within the specified document IDs.

        Args:
            user_id (str): Identifier for the user's Qdrant collection.
            doc_ids (List[int]): List of document IDs to constrain the search.
            string_query (str): Text to vectorize and use as the search query.
            limit (int): Maximum number of results to return. Defaults to 5.
            score_threshold (float): Minimum similarity score for returned results.

        Returns:
            List[Dict]: List of matched chunks with document ID, chunk ID, title, and similarity score.

        Raises:
            ValueError: If the embedding service fails after 5 attempts.
        """
        payload = {
            "text": string_query,
        }
        for _ in range(5):
            embedding_response = self.session.post(
                url = self.embed_str_address,
                json = payload,
                timeout = 10,
            )
            if embedding_response.status_code == 200:
                response_json = embedding_response.json()
                vector = json.loads(response_json["vectorized text"])
                query_response = self.qdrant_handler.search_query_on_doc(
                    user_id = user_id,
                    doc_ids = doc_ids,
                    vector_query = vector,
                    limit = limit,
                    score_threshold = score_threshold,
                )
                logger.info(
                    f"Search query successful for user_id={user_id} on doc_ids={doc_ids}, results={len(query_response)}"
                )
                break
        else:
            error_msg = embedding_response.text
            logger.error(
                f"Failed to vectorize input query for user_id={user_id}:\n{error_msg}"
            )
            raise ValueError(f"Failed to vectorize input query: \n\n{error_msg}")
        return [
            {
                "DocId": r.payload.get("DocId"),
                "ChunkId": r.payload.get("ChunkId"),
                "Title": r.payload.get("Title"),
                "Similarity Score": r.score,
            } for r in query_response
        ]
    
    def scroll_user_collection(
            self,
            user_id: str,
            limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a limited number of vector records from a user's Qdrant collection.

        This method scrolls through the user's collection and returns vector metadata
        including document ID, chunk ID, and title. Useful for previewing or inspecting
        existing data.

        Args:
            user_id (str): The unique identifier for the user.
            limit (int): The maximum number of records to retrieve.

        Returns:
            List[Dict]: A list of dictionaries containing metadata for each vector.
        """
        records = self.qdrant_handler.scroll_collection(
            user_id = user_id,
            limit = limit,
        )
        return [
            {
                "DocId": r.payload.get("DocId"),
                "ChunkId": r.payload.get("ChunkId"),
                "Title": r.payload.get("Title"),
            } for r in records
        ]

    def list_collections(self) -> list[str]:
        """
        Retrieve a list of all collection names currently stored in Qdrant.

        This method delegates to the Qdrant handler and returns the names of all
        collections, typically corresponding to individual users or datasets.

        Returns:
            list[str]: A list of collection names.
        """
        return self.qdrant_handler.list_collections()
    
