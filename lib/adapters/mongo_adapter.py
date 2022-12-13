from pymongo import MongoClient
from bson.objectid import ObjectId


class MongoAdapter(object):
    def __init__(self, host="0.0.0.0", port=27017):
        self.client = MongoClient(host, port)

    def get_db(self, db_name=None):
        try:
            return self.client.get_database(db_name)
        except Exception as e:
            msg = "Error in get_db method"
            raise e(msg)

    def get_collection(self, db_name=None, collection=None):
        try:
            db = self.get_db(db_name)
            coll = db[collection]
            return coll
        except Exception as e:
            msg = "db_name cannot be none"
            raise e(msg)

    def insert_doc(self, doc=None, db_name=None, collection=None, return_id=False):
        try:
            db = self.get_db(db_name)
            inserted_id = db[collection].insert_one(doc).inserted_id
            if return_id:
                return inserted_id
        except Exception as e:
            msg = "Error in insert_doc"
            raise e(msg)

    def bulk_insert_docs(
        self, docs=[], db_name=None, collection=None, return_ids=False
    ):
        try:
            db = self.get_db(db_name)
            inserted_ids = db[collection].insert_many(docs).inserted_ids
            if return_ids:
                return inserted_ids
        except Exception as e:
            msg = "Error in bulk_insert_docs"
            raise e(msg)

    def get_docs_by_match(
        self, db_name=None, collection=None, match_dict={}, greedy=False
    ):
        try:
            db = self.get_db(db_name)
            coll = db[collection]
            if not greedy:
                res = coll.find_one(match_dict)
            else:
                res = list(coll.find(match_dict))
            return res
        except Exception as e:
            msg = "Error in bulk_insert_docs"
            raise e(msg)

    def get_doc_by_id(self, doc_id=None, db_name=None, collection=None):
        try:
            db = self.get_db(db_name)
            if isinstance(doc_id, str):
                doc = db[collection].find_one({"_id": ObjectId(doc_id)})
            elif isinstance(doc_id, ObjectId):
                doc = db[collection].find_one({"_id": doc_id})
            return doc
        except Exception as e:
            msg = "Error in get_doc_by_id"
            raise e(msg)

    def convert_docid_to_str(self, doc_id=None):
        try:
            return str(doc_id)
        except Exception as e:
            msg = "Error in convert_docid_to_str"
            raise e(msg)

    def count_docs_in_collection(self, db_name=None, collection=None, match_dict={}):
        try:
            db = self.get_db(db_name)
            num_docs = db[collection].count_documents(match_dict)
            return num_docs
        except Exception as e:
            msg = "Error in count_docs_in_collection"
            raise e(msg)

    def get_docs_in_range(self, docs=[], db_name=None, collection=None):
        try:
            db = self.get_db(db_name)
            db[collection].insert_many(docs)
        except Exception as e:
            msg = "Error in bulk_insert_docs"
            raise e(msg)

    def update_doc(
        self, db_name=None, collection=None, doc_id=None, col_name=None, new_val=None
    ):
        try:
            db = self.get_db(db_name)
            result = db[collection].find_one_and_update(
                {"_id": ObjectId(doc_id)}, {"$set": {col_name: new_val}}, upsert=True
            )
            if result:
                return result
        except Exception as e:
            msg = "Error in insert_doc"
            raise e(msg)

    def delete_doc_in_collection(self, db_name=None, collection=None, match_dict={}):
        try:
            db = self.get_db(db_name)
            db[collection].delete_one(match_dict)
        except Exception as e:
            msg = "Error in delete_doc_in_collection"
            raise e(msg)

    def delete_docs_in_collection(self, db_name=None, collection=None, match_dict={}):
        try:
            db = self.get_db(db_name)
            db[collection].delete_many(match_dict)
        except Exception as e:
            msg = "Error in delete_docs_in_collection"
            raise e(msg)
