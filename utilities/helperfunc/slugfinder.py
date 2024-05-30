from .dbfunc import perform_database_operation

def get_slug_value(docShorthand):
    query = {"key": docShorthand}
    result = perform_database_operation("test", "defaultslugvalues", "read", query)
    if result:
        return result[0].get("value", None)
    return None