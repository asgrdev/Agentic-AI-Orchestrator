class KuzuBaseException(Exception):
    """Base exception for all KuzuDB errors"""
    pass

class KuzuConnectionError(KuzuBaseException):
    """Raised when connection to database fails"""
    pass

class KuzuQueryError(KuzuBaseException):
    """Raised when a query execution fails"""
    pass

class KuzuNodeNotFoundError(KuzuBaseException):
    """Raised when a node is not found"""
    pass

class KuzuRelationshipError(KuzuBaseException):
    """Raised when relationship operations fail"""
    pass

class KuzuSchemaError(KuzuBaseException):
    """Raised when schema operations fail"""
    pass

class KuzuTransactionError(KuzuBaseException):
    """Raised when transaction operations fail"""
    pass
