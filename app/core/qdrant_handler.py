import uuid
from typing import List, Dict
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
                points = point
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