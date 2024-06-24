from .database import payments_collection

# Helper functions
def create_payment_record(payment_data: dict):
    return payments_collection.insert_one(payment_data).inserted_id

def get_payments(query: dict):
    return list(payments_collection.find(query))