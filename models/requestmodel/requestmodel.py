import json
from typing import List, Optional
from bson import ObjectId
from pydantic import BaseModel

# Pydantic models for request and response
class CourseRequest(BaseModel):
    fos: str
    level: str
    dictionary: dict
    email: str
    LanguageProficiency:str
    Score:float | str
    toggle: bool = False
    partneredtoggle: bool = False
    

class VisaPRRequest(BaseModel):
    email: str
    course: dict
    ask: str

class PromptItem(BaseModel):
    role: str
    content: str

class SOPSOWPRequest(BaseModel):
    prompt: List[PromptItem]
    maxtoken: int 
    model: str

class EncryptedRequest(BaseModel):
    encryptedData: str

class CourseResponse(BaseModel):
    data: dict


class EmailRequest(BaseModel):
    email: str
    Firstname: str
    Lastname: str
    # EmailId: str
    Phone: str
    Password: str
    Second: bool

class ForgetRequest(BaseModel):
    email: str

class AuthRequest(BaseModel):
    authId: str

class ResetPasswordRequest(AuthRequest):
    newpassword:str

class LoginRequest(BaseModel):
    email: str
    Password: str

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
    
class ProfileInfoRequest(BaseModel):
    email: str
    profileInfo: dict

class ProfileInfoResponse(BaseModel):
    responsedata: dict

# Define a custom response model if needed
class ErrorResponseModel(BaseModel):
    detail: str
    err: str

class PhonePeResponse(BaseModel):
    status: str
    message: str

class PhonePeRequest(BaseModel):
    phone: str
    amount: str

class MailAttachment(BaseModel):
    filename: str
    content: str

class AutoMailRequest(BaseModel):
    sender: str
    to: str
    cc:list = None
    subject: str
    attachments: List[MailAttachment]
    name: str = "Aspirant"

class DocuSealRequest(BaseModel):
    email: Optional[str] = None
    serveDoc: str

class CheckDocRequest(DocuSealRequest):
    op: Optional[str] = None