from imports import os, hmac, hashlib

def serialize_payments(payments):
    for payment in payments:
        payment["_id"] = str(payment["_id"])
    return payments

def serialize_payments_with_id(payment):
    if isinstance(payment, dict):
        payment["_id"] = str(payment["_id"])
        return payment
    else:
        raise ValueError("Expected payment to be a dictionary")
    
def generated_signature(razorpay_order_id: str, razorpay_payment_id: str) -> str:
    key_secret = os.getenv("RAZORPAY_API_SECRET")
    if not key_secret:
        raise ValueError("Razorpay key secret is not defined in environment variables.")
    
    message = f"{razorpay_order_id}|{razorpay_payment_id}"
    signature = hmac.new(key_secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    return signature