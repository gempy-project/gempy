class ModelValidationError(ValueError):
    """Raised by GeoModel.validate() when the model is semantically invalid.

    Attributes:
        field: The field or path within the model that caused the failure.
        reason: A short machine-readable reason code (e.g. 'empty_model').
        context: Optional dict with additional diagnostic information.
    """

    def __init__(self, field: str, reason: str, context: dict | None = None):
        self.field = field
        self.reason = reason
        self.context = context or {}
        super().__init__(f"[{field}] {reason}")
